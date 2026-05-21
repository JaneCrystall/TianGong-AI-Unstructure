"""KB parse and S3-ready worker orchestration."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import psycopg2
import requests

from . import control_plane, queue
from .artifacts import write_processed_artifacts
from .chunk_splitter import split_chunks_for_embedding
from .config import WorkerConfig
from .embedding_client import EmbeddingError, add_chunk_embeddings
from .manifest import load_artifact_info
from .parser_adapter import (
    ParserError,
    ParserTaskFailure,
    ParserTimeout,
    fetch_two_stage_queue_status,
    fetch_two_stage_task_status,
    parse_with_unstructure_serve,
    submit_two_stage_task,
)
from .s3_ready import processed_manifest_key, wait_for_s3_processed_ready
from .snapshot import (
    load_parse_snapshot,
    load_s3_ready_snapshot,
    resolve_raw_path,
    validate_raw_storage_path,
)

LOGGER = logging.getLogger(__name__)
FINALIZE_DB_MAX_ATTEMPTS = 3
FINALIZE_DB_INITIAL_BACKOFF_SECONDS = 1.0


class JobTimeout(RuntimeError):
    pass


class JobDeadline:
    def __init__(self, stage: str, timeout_seconds: int):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        self.deadline = time.monotonic() + timeout_seconds

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise JobTimeout(f"{self.stage.upper()}_JOB_TIMEOUT_AFTER_{self.timeout_seconds}s")

    def remaining_seconds(self, maximum: int | None = None) -> int:
        self.check()
        remaining = max(1, int(self.deadline - time.monotonic()))
        if maximum is not None:
            return max(1, min(maximum, remaining))
        return remaining


def _http_status_from_message(prefix: str, message: str) -> int | None:
    if not message.startswith(prefix):
        return None
    try:
        return int(message.removeprefix(prefix).split(":", 1)[0].strip())
    except ValueError:
        return None


def _is_retryable_http_status(status: int | None) -> bool:
    return status is not None and (status == 429 or status >= 500)


def is_parse_failure_retryable(error: Exception) -> bool:
    if isinstance(error, JobTimeout):
        return True
    if isinstance(error, requests.RequestException):
        return True

    message = str(error)
    if message in {"RAW_HASH_MISMATCH", "EMPTY_RESULT", "EMBEDDING_CHUNK_NOT_OBJECT"}:
        return False
    if message.startswith("RAW_STORAGE_PATH_MISMATCH"):
        return False
    if message.startswith("EMBEDDING_DIMENSION_TOO_SMALL"):
        return False
    if message.startswith("parser response ") or message.startswith("parser result "):
        return False
    if message.startswith("embedding response "):
        return False

    if isinstance(error, ParserTimeout):
        return True
    if isinstance(error, ParserTaskFailure):
        return True
    if isinstance(error, ParserError):
        return _is_retryable_http_status(_http_status_from_message("parser http error ", message))
    if isinstance(error, EmbeddingError):
        status = _http_status_from_message("embedding http error ", message)
        return status is None or _is_retryable_http_status(status)

    return True


def is_s3_ready_failure_retryable(error: Exception) -> bool:
    if isinstance(error, JobTimeout):
        return True

    message = str(error)
    if message in {"LOCAL_MANIFEST_MISSING"}:
        return False
    if "S3_MANIFEST_MISMATCH" in message:
        return False
    if "S3_ARTIFACT_MISMATCH" in message and "sha256" in message:
        return False

    return True


def archive_current_message(conn, queue_name: str, msg_id: int) -> bool:
    return queue.archive_job_message_by_id(conn, queue_name, msg_id)


def read_queue_message(
    config: WorkerConfig,
    queue_name: str,
) -> queue.QueueMessage | None:
    with control_plane.connect(config.database_url) as conn:
        return queue.read_one(conn, queue_name, config.queue_vt_seconds)


def claim_queue_message(
    config: WorkerConfig,
    queue_name: str,
    message: queue.QueueMessage,
) -> control_plane.ClaimJobResult | None:
    with control_plane.connect(config.database_url) as conn:
        claim = control_plane.claim_job(
            conn,
            message.job_id,
            queue_name,
            message.msg_id,
            config.worker_id,
            config.lock_seconds,
        )
        if claim is None or not claim.claimed:
            handle_unclaimed_message(conn, queue_name, message, claim)
        return claim


def load_parse_snapshot_with_short_connection(
    config: WorkerConfig,
    job_id: str,
):
    with control_plane.connect(config.database_url) as conn:
        return load_parse_snapshot(conn, job_id)


def load_s3_ready_snapshot_with_short_connection(
    config: WorkerConfig,
    job_id: str,
):
    with control_plane.connect(config.database_url) as conn:
        return load_s3_ready_snapshot(conn, job_id)


def _is_retryable_finalization_error(error: Exception) -> bool:
    return isinstance(error, (psycopg2.InterfaceError, psycopg2.OperationalError))


def _finalization_backoff_seconds(attempt: int) -> float:
    return FINALIZE_DB_INITIAL_BACKOFF_SECONDS * (2 ** max(0, attempt - 1))


def complete_parse_local_ready_and_archive_with_retry(
    config: WorkerConfig,
    job_id: str,
    document_id: str,
    document_version: int,
    manifest_local_uri: str,
    artifact_uuid: str,
    manifest_hash: str,
    chunk_count: int,
    metadata_json: dict,
    s3_ready_payload_json: dict,
    queue_name: str,
    msg_id: int,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> control_plane.S3ReadyEnqueueResult:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                result = control_plane.complete_parse_local_ready_and_enqueue_s3_check(
                    active_conn,
                    job_id,
                    config.worker_id,
                    document_id,
                    document_version,
                    manifest_local_uri,
                    artifact_uuid,
                    manifest_hash,
                    chunk_count,
                    metadata_json,
                    s3_ready_payload_json,
                )
                if result is None:
                    raise RuntimeError(
                        "complete_parse_local_ready_and_enqueue_s3_check returned no row"
                    )
                archive_current_message(active_conn, queue_name, msg_id)
                return result
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "parse job %s finalization DB write failed on attempt %s/%s; "
                "reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("parse finalization retry loop exhausted")


def complete_parse_local_ready_with_retry(
    config: WorkerConfig,
    job_id: str,
    document_id: str,
    document_version: int,
    manifest_local_uri: str,
    artifact_uuid: str,
    manifest_hash: str,
    chunk_count: int,
    metadata_json: dict,
    s3_ready_payload_json: dict,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> control_plane.S3ReadyEnqueueResult:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                result = control_plane.complete_parse_local_ready_and_enqueue_s3_check(
                    active_conn,
                    job_id,
                    config.worker_id,
                    document_id,
                    document_version,
                    manifest_local_uri,
                    artifact_uuid,
                    manifest_hash,
                    chunk_count,
                    metadata_json,
                    s3_ready_payload_json,
                )
                if result is None:
                    raise RuntimeError(
                        "complete_parse_local_ready_and_enqueue_s3_check returned no row"
                    )
                return result
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "parse job %s async finalization DB write failed on attempt %s/%s; "
                "reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("parse async finalization retry loop exhausted")


def register_parse_two_stage_task_and_archive_with_retry(
    config: WorkerConfig,
    job_id: str,
    task_id: str,
    task_url: str,
    status_url: str,
    queue_name: str,
    msg_id: int,
    backlog_json: dict,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> control_plane.TwoStageRegistrationResult:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                result = control_plane.register_parse_two_stage_task(
                    active_conn,
                    job_id,
                    config.worker_id,
                    task_id,
                    task_url,
                    status_url,
                    queue_name,
                    msg_id,
                    config.lock_seconds,
                    backlog_json,
                )
                if result is None:
                    raise RuntimeError("register_parse_two_stage_task returned no row")
                return result
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "parse job %s two-stage registration DB write failed on attempt %s/%s; "
                "reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("two-stage registration retry loop exhausted")


def handle_unclaimed_message(
    conn,
    queue_name: str,
    message: queue.QueueMessage,
    claim: control_plane.ClaimJobResult | None,
) -> None:
    if claim is None:
        LOGGER.warning("job %s returned no claim disposition", message.job_id)
        return
    LOGGER.info(
        "job %s was not claimed status=%s archive_current_message=%s next_retry_at=%s retry_wakeup_msg_id=%s",
        message.job_id,
        claim.claim_status,
        claim.archive_current_message,
        claim.next_retry_at,
        claim.retry_wakeup_msg_id,
    )
    if claim.archive_current_message:
        archive_current_message(conn, queue_name, message.msg_id)


def fail_job_and_archive_current_message(
    config: WorkerConfig,
    job_id: str,
    queue_name: str,
    msg_id: int,
    worker_id: str,
    retryable: bool,
    error: str,
    error_stage: str,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> control_plane.FailJobResult | None:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                result = control_plane.fail_job(
                    active_conn,
                    job_id,
                    worker_id,
                    retryable,
                    error,
                    error_stage,
                )
                if result is not None:
                    LOGGER.info(
                        "job %s failed status=%s retryable=%s next_retry_at=%s retry_wakeup_msg_id=%s",
                        job_id,
                        result.job_status,
                        retryable,
                        result.next_retry_at,
                        result.retry_wakeup_msg_id,
                    )
                    archive_current_message(active_conn, queue_name, msg_id)
                return result
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "job %s fail_job DB write failed on attempt %s/%s; reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("fail_job retry loop exhausted")


def fail_job_with_retry(
    config: WorkerConfig,
    job_id: str,
    worker_id: str,
    retryable: bool,
    error: str,
    error_stage: str,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> control_plane.FailJobResult | None:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                result = control_plane.fail_job(
                    active_conn,
                    job_id,
                    worker_id,
                    retryable,
                    error,
                    error_stage,
                )
                if result is not None:
                    LOGGER.info(
                        "job %s failed status=%s retryable=%s next_retry_at=%s retry_wakeup_msg_id=%s",
                        job_id,
                        result.job_status,
                        retryable,
                        result.next_retry_at,
                        result.retry_wakeup_msg_id,
                    )
                return result
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "job %s fail_job DB write failed on attempt %s/%s; reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("fail_job retry loop exhausted")


def complete_s3_ready_check_and_archive_with_retry(
    config: WorkerConfig,
    job_id: str,
    document_id: str,
    document_version: int,
    manifest_s3_key: str,
    manifest_hash: str,
    artifact_uuid: str,
    manifest_local_uri: str,
    chunk_count: int,
    queue_name: str,
    msg_id: int,
    max_attempts: int = FINALIZE_DB_MAX_ATTEMPTS,
) -> bool:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    for attempt in range(1, max_attempts + 1):
        try:
            with control_plane.connect(config.database_url) as active_conn:
                ok = control_plane.complete_s3_ready_check(
                    active_conn,
                    job_id,
                    config.worker_id,
                    document_id,
                    document_version,
                    manifest_s3_key,
                    manifest_hash,
                    artifact_uuid,
                    manifest_local_uri,
                    chunk_count,
                )
                if not ok:
                    raise RuntimeError("complete_s3_ready_check returned false")
                archive_current_message(active_conn, queue_name, msg_id)
                return True
        except Exception as exc:
            if not _is_retryable_finalization_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "s3_ready job %s finalization DB write failed on attempt %s/%s; "
                "reconnecting before retry",
                job_id,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(_finalization_backoff_seconds(attempt))

    raise RuntimeError("s3_ready finalization retry loop exhausted")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw_file(path: Path, expected_size: int | None, expected_sha256: str) -> None:
    if not path.exists():
        raise RuntimeError("RAW_NOT_FOUND")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError("RAW_SIZE_MISMATCH")
    if _sha256(path).lower() != expected_sha256.lower():
        raise RuntimeError("RAW_HASH_MISMATCH")


def build_s3_ready_payload(config: WorkerConfig, snapshot) -> dict:
    return {
        "collection_path": snapshot.collection_path,
        "collection_storage_path": snapshot.collection_storage_path,
        "processed_storage_path": snapshot.processed_storage_path,
        "manifest_s3_key": processed_manifest_key(
            config.s3_processed_prefix,
            snapshot,
        ),
        "s3_bucket": config.s3_bucket,
        "s3_prefix": config.s3_processed_prefix,
    }


def write_processed_artifacts_from_parsed(
    config: WorkerConfig,
    job_id: str,
    snapshot,
    parsed,
    deadline: JobDeadline,
    lease=None,
) -> tuple[str, object, dict]:
    result = parsed.result
    if parsed.dropped_empty_text_count:
        LOGGER.info(
            "parse job %s dropped %s empty text chunk(s) from %s parser chunk(s)",
            job_id,
            parsed.dropped_empty_text_count,
            parsed.original_chunk_count,
        )
    if lease is not None:
        lease.check()
    deadline.check()

    embedding_chunks, split_stats = split_chunks_for_embedding(
        result,
        snapshot.document_id,
        config.embedding_chunk_max_tokens,
    )
    if split_stats.split_parent_count:
        LOGGER.info(
            "parse job %s split %s/%s parser chunk(s) into %s embedding chunk(s)",
            job_id,
            split_stats.split_parent_count,
            split_stats.source_chunk_count,
            split_stats.output_chunk_count,
        )
    if lease is not None:
        lease.check()
    deadline.check()

    embedded_result = add_chunk_embeddings(
        embedding_chunks,
        config.embedding_base_url,
        config.embedding_model,
        config.embedding_api_key,
        config.embedding_dimensions,
        config.embedding_batch_size,
        deadline.remaining_seconds(config.embedding_timeout_seconds),
        deadline.check,
    )
    if lease is not None:
        lease.check()
    deadline.check()

    embedding_metadata = {
        "model": config.embedding_model,
        "base_url": config.embedding_base_url,
        "dimensions": config.embedding_dimensions,
        "normalized": True,
        "source_dimensions": "provider_default",
        "chunk_max_tokens": config.embedding_chunk_max_tokens,
        "source_chunk_count": split_stats.source_chunk_count,
        "embedding_chunk_count": split_stats.output_chunk_count,
        "split_parent_count": split_stats.split_parent_count,
    }
    final_dir, artifact_info = write_processed_artifacts(
        embedded_result,
        snapshot,
        config.nas_processed_root,
        config.parser_profile,
        config.parser_version,
        embedding_metadata,
        parsed.txt,
        result,
    )
    if lease is not None:
        lease.check()
    deadline.check()

    manifest_local_uri = final_dir.joinpath("manifest.json").as_posix()
    metadata_json = {
        "processed": {
            "parser_profile": config.parser_profile,
            "parser_version": config.parser_version,
            "chunk_count": artifact_info.chunk_count,
            "artifact_uuid": artifact_info.artifact_uuid,
            "source_chunk_count": parsed.original_chunk_count,
            "dropped_empty_text_count": parsed.dropped_empty_text_count,
            "embedding": embedding_metadata,
        }
    }
    return manifest_local_uri, artifact_info, metadata_json


class LeaseMaintainer:
    def __init__(self, config: WorkerConfig, job_id: str):
        self.config = config
        self.job_id = job_id
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"kb-parse-heartbeat-{job_id}",
            daemon=True,
        )

    def __enter__(self) -> "LeaseMaintainer":
        self.heartbeat()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def heartbeat(self) -> None:
        with control_plane.connect(self.config.database_url) as conn:
            ok = control_plane.heartbeat_job(
                conn,
                self.job_id,
                self.config.worker_id,
                self.config.lock_seconds,
                self.config.queue_vt_seconds,
            )
        if not ok:
            raise RuntimeError("HEARTBEAT_LOST")

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("HEARTBEAT_LOST") from self._error

    def _run(self) -> None:
        while not self._stop.wait(self.config.heartbeat_interval_seconds):
            try:
                self.heartbeat()
            except Exception as exc:
                LOGGER.exception("heartbeat failed for parse job %s", self.job_id)
                self._error = exc
                self._stop.set()


class ParseWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def run_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(self.config.poll_interval_seconds)

    def run_once(self) -> bool:
        message = read_queue_message(self.config, self.config.queue_name)
        if message is None:
            return False
        self.process_message(message)
        return True

    def process_message(self, message: queue.QueueMessage) -> None:
        claimed = claim_queue_message(self.config, self.config.queue_name, message)
        if claimed is None or not claimed.claimed:
            return
        if claimed.stage != "parse":
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.queue_name,
                message.msg_id,
                self.config.worker_id,
                False,
                "UNSUPPORTED_STAGE",
                "parse",
            )
            return

        try:
            with LeaseMaintainer(self.config, claimed.job_id) as lease:
                deadline = JobDeadline("parse", self.config.parse_job_timeout_seconds)
                deadline.check()
                snapshot = load_parse_snapshot_with_short_connection(
                    self.config,
                    claimed.job_id,
                )
                if snapshot.processed_manifest_local_uri and snapshot.processed_artifact_uuid:
                    manifest_path = Path(snapshot.processed_manifest_local_uri)
                    artifact_info = load_artifact_info(manifest_path, snapshot.processed_manifest_hash)
                    manifest_local_uri = manifest_path.as_posix()
                    metadata_json = {
                        "processed": {
                            "parser_profile": artifact_info.manifest.get(
                                "parser_profile",
                                self.config.parser_profile,
                            ),
                            "parser_version": artifact_info.manifest.get(
                                "parser_version",
                                self.config.parser_version,
                            ),
                            "chunk_count": artifact_info.chunk_count,
                            "artifact_uuid": artifact_info.artifact_uuid,
                        }
                    }
                else:
                    deadline.check()
                    raw_path = resolve_raw_path(snapshot.raw_uri, self.config.nas_raw_root)
                    validate_raw_storage_path(snapshot, raw_path, self.config.nas_raw_root)
                    _validate_raw_file(raw_path, snapshot.file_size, snapshot.sha256)
                    lease.check()
                    deadline.check()

                    parsed = parse_with_unstructure_serve(
                        raw_path,
                        self.config.unstructure_serve_url,
                        self.config.unstructure_serve_bearer_token,
                        timeout_seconds=deadline.remaining_seconds(
                            self.config.parse_job_timeout_seconds
                        ),
                        use_two_stage=self.config.use_two_stage_parser,
                        two_stage_base_url=self.config.unstructure_serve_two_stage_base_url,
                        two_stage_submit_timeout_seconds=(
                            self.config.two_stage_submit_timeout_seconds
                        ),
                        two_stage_status_timeout_seconds=(
                            self.config.two_stage_status_timeout_seconds
                        ),
                        two_stage_poll_interval_seconds=(
                            self.config.two_stage_poll_interval_seconds
                        ),
                        two_stage_priority=self.config.two_stage_priority,
                        two_stage_chunk_type=True,
                        two_stage_provider=self.config.two_stage_provider,
                        two_stage_model=self.config.two_stage_model,
                        two_stage_prompt=self.config.two_stage_prompt,
                    )
                    manifest_local_uri, artifact_info, metadata_json = (
                        write_processed_artifacts_from_parsed(
                            self.config,
                            claimed.job_id,
                            snapshot,
                            parsed,
                            deadline,
                            lease,
                        )
                    )

                s3_ready_payload_json = build_s3_ready_payload(self.config, snapshot)
                s3_ready_result = complete_parse_local_ready_and_archive_with_retry(
                    self.config,
                    claimed.job_id,
                    claimed.document_id,
                    claimed.document_version,
                    manifest_local_uri,
                    artifact_info.artifact_uuid,
                    artifact_info.manifest_hash,
                    artifact_info.chunk_count,
                    metadata_json,
                    s3_ready_payload_json,
                    self.config.queue_name,
                    message.msg_id,
                )
                LOGGER.info(
                    "parse job %s completed local artifacts and queued s3_ready job %s msg %s",
                    claimed.job_id,
                    s3_ready_result.s3_ready_job_id,
                    s3_ready_result.s3_ready_msg_id,
                )
        except Exception as exc:
            LOGGER.exception("parse job %s failed", claimed.job_id)
            retryable = is_parse_failure_retryable(exc)
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.queue_name,
                message.msg_id,
                self.config.worker_id,
                retryable,
                str(exc),
                "parse",
            )


class ParseSubmitterWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def run_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(self.config.poll_interval_seconds)

    def run_once(self) -> bool:
        if not self.config.use_two_stage_parser:
            raise RuntimeError("parse-submitter requires KB_PARSE_USE_TWO_STAGE=true")
        try:
            queue_status = fetch_two_stage_queue_status(
                self.config.unstructure_serve_two_stage_base_url,
                self.config.unstructure_serve_bearer_token,
                timeout_seconds=self.config.two_stage_queue_status_timeout_seconds,
            )
        except Exception:
            LOGGER.warning("could not fetch two-stage queue status; submitter is backing off", exc_info=True)
            return False

        backlog = queue_status.backlog(self.config.two_stage_parse_queue_name)
        if backlog >= self.config.two_stage_max_parse_backlog:
            LOGGER.info(
                "two-stage parse backlog %s >= limit %s; submitter is backing off",
                backlog,
                self.config.two_stage_max_parse_backlog,
            )
            return False

        message = read_queue_message(self.config, self.config.queue_name)
        if message is None:
            return False
        self.process_message(message, queue_status.queues, queue_status.unacked)
        return True

    def process_message(
        self,
        message: queue.QueueMessage,
        queues: dict[str, int],
        unacked: dict[str, int],
    ) -> None:
        claimed = claim_queue_message(self.config, self.config.queue_name, message)
        if claimed is None or not claimed.claimed:
            return
        if claimed.stage != "parse":
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.queue_name,
                message.msg_id,
                self.config.worker_id,
                False,
                "UNSUPPORTED_STAGE",
                "parse",
            )
            return

        submitted = None
        try:
            snapshot = load_parse_snapshot_with_short_connection(self.config, claimed.job_id)
            raw_path = resolve_raw_path(snapshot.raw_uri, self.config.nas_raw_root)
            validate_raw_storage_path(snapshot, raw_path, self.config.nas_raw_root)
            _validate_raw_file(raw_path, snapshot.file_size, snapshot.sha256)

            submitted = submit_two_stage_task(
                raw_path,
                self.config.unstructure_serve_two_stage_base_url,
                self.config.unstructure_serve_bearer_token,
                timeout_seconds=self.config.two_stage_submit_timeout_seconds,
                return_txt=True,
                priority=self.config.two_stage_priority,
                chunk_type=True,
                provider=self.config.two_stage_provider,
                model=self.config.two_stage_model,
                prompt=self.config.two_stage_prompt,
            )
            registered = register_parse_two_stage_task_and_archive_with_retry(
                self.config,
                claimed.job_id,
                submitted.task_id,
                submitted.task_url,
                submitted.status_url,
                self.config.queue_name,
                message.msg_id,
                {
                    "queues": queues,
                    "unacked": unacked,
                    "parse_queue": self.config.two_stage_parse_queue_name,
                },
            )
            LOGGER.info(
                "parse job %s submitted two-stage task %s document=%s status=%s",
                claimed.job_id,
                registered.task_id,
                registered.document_id,
                registered.document_status,
            )
        except Exception as exc:
            LOGGER.exception("parse job %s two-stage submission failed", claimed.job_id)
            if submitted is not None:
                LOGGER.error(
                    "parse job %s submitted two-stage task %s but did not persist registration",
                    claimed.job_id,
                    submitted.task_id,
                )
                return
            retryable = is_parse_failure_retryable(exc)
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.queue_name,
                message.msg_id,
                self.config.worker_id,
                retryable,
                str(exc),
                "parse",
            )


class ParseFinalizerWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config.poll_interval_seconds)

    def run_once(self) -> bool:
        if not self.config.use_two_stage_parser:
            raise RuntimeError("parse-finalizer requires KB_PARSE_USE_TWO_STAGE=true")
        processed = False
        for _ in range(self.config.two_stage_finalizer_limit):
            claimed = self.claim_next()
            if claimed is None:
                break
            self.process_claim(claimed)
            processed = True
        return processed

    def claim_next(self) -> control_plane.TwoStageFinalizationClaim | None:
        with control_plane.connect(self.config.database_url) as conn:
            return control_plane.claim_parse_two_stage_task(
                conn,
                self.config.worker_id,
                self.config.lock_seconds,
            )

    def mark_state(
        self,
        job_id: str,
        state: str,
        error: str | None = None,
    ) -> None:
        with control_plane.connect(self.config.database_url) as conn:
            ok = control_plane.mark_parse_two_stage_task_state(
                conn,
                job_id,
                self.config.worker_id,
                state,
                error,
                max(1, self.config.poll_interval_seconds * 2),
            )
        if not ok:
            raise RuntimeError("mark_parse_two_stage_task_state returned false")

    def process_claim(self, claimed: control_plane.TwoStageFinalizationClaim) -> None:
        try:
            try:
                status = fetch_two_stage_task_status(
                    self.config.unstructure_serve_two_stage_base_url,
                    self.config.unstructure_serve_bearer_token,
                    claimed.task_id,
                    timeout_seconds=self.config.two_stage_status_timeout_seconds,
                    return_txt=True,
                )
            except Exception as exc:
                LOGGER.warning(
                    "parse job %s two-stage task %s status poll failed; will retry",
                    claimed.job_id,
                    claimed.task_id,
                    exc_info=True,
                )
                self.mark_state(claimed.job_id, "POLL_ERROR", str(exc))
                return

            if status.state in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
                self.mark_state(claimed.job_id, status.state)
                return
            if status.state in {"FAILURE", "REVOKED"}:
                error = status.error or status.state
                self.mark_state(claimed.job_id, status.state, error)
                raise ParserTaskFailure(f"parser two-stage task {claimed.task_id} failed: {error}")
            if status.state != "SUCCESS" or status.parsed is None:
                self.mark_state(claimed.job_id, status.state)
                return

            snapshot = load_parse_snapshot_with_short_connection(self.config, claimed.job_id)
            deadline = JobDeadline("parse", self.config.parse_job_timeout_seconds)
            if snapshot.processed_manifest_local_uri and snapshot.processed_artifact_uuid:
                manifest_path = Path(snapshot.processed_manifest_local_uri)
                artifact_info = load_artifact_info(
                    manifest_path,
                    snapshot.processed_manifest_hash,
                )
                manifest_local_uri = manifest_path.as_posix()
                metadata_json = {
                    "processed": {
                        "parser_profile": artifact_info.manifest.get(
                            "parser_profile",
                            self.config.parser_profile,
                        ),
                        "parser_version": artifact_info.manifest.get(
                            "parser_version",
                            self.config.parser_version,
                        ),
                        "chunk_count": artifact_info.chunk_count,
                        "artifact_uuid": artifact_info.artifact_uuid,
                    }
                }
            else:
                manifest_local_uri, artifact_info, metadata_json = (
                    write_processed_artifacts_from_parsed(
                        self.config,
                        claimed.job_id,
                        snapshot,
                        status.parsed,
                        deadline,
                    )
                )

            result = complete_parse_local_ready_with_retry(
                self.config,
                claimed.job_id,
                claimed.document_id,
                claimed.document_version,
                manifest_local_uri,
                artifact_info.artifact_uuid,
                artifact_info.manifest_hash,
                artifact_info.chunk_count,
                metadata_json,
                build_s3_ready_payload(self.config, snapshot),
            )
            LOGGER.info(
                "parse job %s finalized two-stage task %s and queued s3_ready job %s msg %s",
                claimed.job_id,
                claimed.task_id,
                result.s3_ready_job_id,
                result.s3_ready_msg_id,
            )
        except Exception as exc:
            LOGGER.exception("parse job %s two-stage finalization failed", claimed.job_id)
            retryable = is_parse_failure_retryable(exc)
            fail_job_with_retry(
                self.config,
                claimed.job_id,
                self.config.worker_id,
                retryable,
                str(exc),
                "parse",
            )


class S3ReadyWorker:
    def __init__(self, config: WorkerConfig):
        self.config = config

    def run_forever(self) -> None:
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(self.config.poll_interval_seconds)

    def run_once(self) -> bool:
        message = read_queue_message(self.config, self.config.s3_ready_queue_name)
        if message is None:
            return False
        self.process_message(message)
        return True

    def process_message(self, message: queue.QueueMessage) -> None:
        claimed = claim_queue_message(self.config, self.config.s3_ready_queue_name, message)
        if claimed is None or not claimed.claimed:
            return
        if claimed.stage != "s3_ready":
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.s3_ready_queue_name,
                message.msg_id,
                self.config.worker_id,
                False,
                "UNSUPPORTED_STAGE",
                "s3_ready",
            )
            return

        try:
            with LeaseMaintainer(self.config, claimed.job_id) as lease:
                deadline = JobDeadline("s3_ready", self.config.s3_ready_job_timeout_seconds)
                deadline.check()
                snapshot = load_s3_ready_snapshot_with_short_connection(
                    self.config,
                    claimed.job_id,
                )
                if not snapshot.processed_manifest_local_uri:
                    raise RuntimeError("LOCAL_MANIFEST_MISSING")
                manifest_path = Path(snapshot.processed_manifest_local_uri)
                artifact_info = load_artifact_info(manifest_path, snapshot.processed_manifest_hash)
                lease.check()
                deadline.check()

                if self.config.s3_ready_mode == "skip":
                    manifest_s3_key = processed_manifest_key(self.config.s3_processed_prefix, snapshot)
                else:
                    if not self.config.s3_bucket:
                        raise RuntimeError("S3_NOT_READY: KB_PROCESSED_S3_BUCKET is required")
                    ready = wait_for_s3_processed_ready(
                        snapshot,
                        artifact_info,
                        self.config.s3_bucket,
                        self.config.s3_processed_prefix,
                        self.config.s3_strict_hash,
                        deadline.remaining_seconds(self.config.s3_ready_timeout_seconds),
                        self.config.s3_ready_poll_interval_seconds,
                    )
                    manifest_s3_key = ready.manifest_s3_key
                lease.check()
                deadline.check()

                ok = complete_s3_ready_check_and_archive_with_retry(
                    self.config,
                    claimed.job_id,
                    claimed.document_id,
                    claimed.document_version,
                    manifest_s3_key,
                    artifact_info.manifest_hash,
                    artifact_info.artifact_uuid,
                    manifest_path.as_posix(),
                    artifact_info.chunk_count,
                    self.config.s3_ready_queue_name,
                    message.msg_id,
                )
                if not ok:
                    raise RuntimeError("complete_s3_ready_check returned false")
        except Exception as exc:
            LOGGER.exception("s3_ready job %s failed", claimed.job_id)
            retryable = is_s3_ready_failure_retryable(exc)
            fail_job_and_archive_current_message(
                self.config,
                claimed.job_id,
                self.config.s3_ready_queue_name,
                message.msg_id,
                self.config.worker_id,
                retryable,
                str(exc),
                "s3_ready",
            )
