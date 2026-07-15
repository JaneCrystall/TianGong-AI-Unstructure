from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import psycopg2

from src.kb_parse_worker import control_plane, queue
from src.kb_parse_worker.worker import (
    JobTimeout,
    ParseWorker,
    S3ReadyWorker,
    complete_s3_ready_check_and_archive_with_retry,
    complete_parse_local_ready_and_archive_with_retry,
    fail_job_and_archive_current_message,
    handle_unclaimed_message,
    is_parse_failure_retryable,
    is_s3_ready_failure_retryable,
)


class NoopLease:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "NoopLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def check(self) -> None:
        return None


def worker_config() -> SimpleNamespace:
    return SimpleNamespace(
        database_url="postgresql://test",
        worker_id="worker-1",
        queue_name="kb_parse_queue",
        s3_ready_queue_name="kb_s3_ready_queue",
        queue_vt_seconds=30,
        lock_seconds=30,
        heartbeat_interval_seconds=1,
        poll_interval_seconds=1,
        parse_job_timeout_seconds=60,
        s3_ready_job_timeout_seconds=60,
        s3_ready_mode="check",
        s3_bucket="tiangong",
        s3_processed_prefix="processed_docs",
        s3_strict_hash=False,
        s3_ready_timeout_seconds=30,
        s3_ready_poll_interval_seconds=1,
    )


def claimed(stage: str) -> control_plane.ClaimJobResult:
    return control_plane.ClaimJobResult(
        claim_status="claimed",
        archive_current_message=False,
        job_id="job-1",
        document_id="00000000-0000-0000-0000-000000000001",
        document_version=1,
        stage=stage,
        status="running",
        payload_json={},
        next_retry_at=None,
        retry_wakeup_msg_id=None,
    )


def claim_disposition(status: str, archive_current_message: bool) -> control_plane.ClaimJobResult:
    return control_plane.ClaimJobResult(
        claim_status=status,
        archive_current_message=archive_current_message,
        job_id="job-1",
        document_id="00000000-0000-0000-0000-000000000001",
        document_version=1,
        stage="parse",
        status="failed",
        payload_json={},
        next_retry_at="2026-05-11T10:00:00+00:00",
        retry_wakeup_msg_id=42,
    )


def message() -> queue.QueueMessage:
    return queue.QueueMessage(msg_id=1, job_id="job-1", raw_payload={"job_id": "job-1"})


class FakeConnection:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None


class KbParseWorkerReliabilityTests(unittest.TestCase):
    def test_idle_poll_backs_off_and_caps(self) -> None:
        config = worker_config()
        config.poll_interval_seconds = 1
        config.idle_poll_interval_max_seconds = 4
        worker = ParseWorker(config)

        with (
            patch.object(worker, "run_once", return_value=False),
            patch(
                "src.kb_parse_worker.worker.time.sleep",
                side_effect=[None, None, None, KeyboardInterrupt],
            ) as sleep,
        ):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_forever()

        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(4), call(4)])

    def test_idle_poll_resets_after_processed_work(self) -> None:
        config = worker_config()
        config.poll_interval_seconds = 1
        config.idle_poll_interval_max_seconds = 4
        worker = ParseWorker(config)

        with (
            patch.object(worker, "run_once", side_effect=[False, False, True, False]),
            patch(
                "src.kb_parse_worker.worker.time.sleep",
                side_effect=[None, None, KeyboardInterrupt],
            ) as sleep,
        ):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_forever()

        self.assertEqual(sleep.call_args_list, [call(1), call(2), call(1)])

    def test_parse_failure_classifier_distinguishes_terminal_and_transient(self) -> None:
        self.assertFalse(is_parse_failure_retryable(RuntimeError("EMPTY_RESULT")))
        self.assertFalse(is_parse_failure_retryable(RuntimeError("RAW_STORAGE_PATH_MISMATCH: path")))
        self.assertTrue(is_parse_failure_retryable(RuntimeError("temporary parser outage")))
        self.assertTrue(is_parse_failure_retryable(JobTimeout("PARSE_JOB_TIMEOUT_AFTER_60s")))

    def test_s3_ready_failure_classifier_distinguishes_terminal_and_transient(self) -> None:
        self.assertFalse(is_s3_ready_failure_retryable(RuntimeError("LOCAL_MANIFEST_MISSING")))
        self.assertFalse(
            is_s3_ready_failure_retryable(RuntimeError("S3_MANIFEST_MISMATCH: document_id"))
        )
        self.assertFalse(
            is_s3_ready_failure_retryable(RuntimeError("S3_ARTIFACT_MISMATCH: chunks.pkl sha256"))
        )
        self.assertTrue(
            is_s3_ready_failure_retryable(RuntimeError("S3_NOT_READY_AFTER_TIMEOUT: not synced"))
        )
        self.assertTrue(is_s3_ready_failure_retryable(JobTimeout("S3_READY_JOB_TIMEOUT_AFTER_60s")))

    def test_parse_terminal_failure_marks_non_retryable_and_archives_dead_job(self) -> None:
        config = worker_config()
        with (
            patch("src.kb_parse_worker.worker.LeaseMaintainer", NoopLease),
            patch("src.kb_parse_worker.worker.claim_queue_message", return_value=claimed("parse")),
            patch(
                "src.kb_parse_worker.worker.load_parse_snapshot_with_short_connection",
                side_effect=RuntimeError("EMPTY_RESULT"),
            ),
            patch(
                "src.kb_parse_worker.worker.fail_job_and_archive_current_message",
            ) as fail_job,
            patch("src.kb_parse_worker.worker.LOGGER.exception"),
        ):
            ParseWorker(config).process_message(message())

        fail_job.assert_called_once()
        self.assertEqual(fail_job.call_args.args[1:5], ("job-1", "kb_parse_queue", 1, "worker-1"))
        self.assertFalse(fail_job.call_args.args[5])
        self.assertEqual(fail_job.call_args.args[7], "parse")

    def test_parse_timeout_marks_retryable_and_archives_current_message(self) -> None:
        config = worker_config()
        with (
            patch("src.kb_parse_worker.worker.LeaseMaintainer", NoopLease),
            patch("src.kb_parse_worker.worker.claim_queue_message", return_value=claimed("parse")),
            patch(
                "src.kb_parse_worker.worker.load_parse_snapshot_with_short_connection",
                side_effect=JobTimeout("PARSE_JOB_TIMEOUT_AFTER_60s"),
            ),
            patch(
                "src.kb_parse_worker.worker.fail_job_and_archive_current_message",
            ) as fail_job,
            patch("src.kb_parse_worker.worker.LOGGER.exception"),
        ):
            ParseWorker(config).process_message(message())

        fail_job.assert_called_once()
        self.assertTrue(fail_job.call_args.args[5])
        self.assertEqual(fail_job.call_args.args[7], "parse")

    def test_s3_ready_terminal_failure_marks_non_retryable_and_archives_dead_job(self) -> None:
        config = worker_config()
        with (
            patch("src.kb_parse_worker.worker.LeaseMaintainer", NoopLease),
            patch("src.kb_parse_worker.worker.claim_queue_message", return_value=claimed("s3_ready")),
            patch(
                "src.kb_parse_worker.worker.load_s3_ready_snapshot_with_short_connection",
                return_value=SimpleNamespace(processed_manifest_local_uri=None),
            ),
            patch(
                "src.kb_parse_worker.worker.fail_job_and_archive_current_message",
            ) as fail_job,
            patch("src.kb_parse_worker.worker.LOGGER.exception"),
        ):
            S3ReadyWorker(config).process_message(message())

        fail_job.assert_called_once()
        self.assertEqual(fail_job.call_args.args[1:5], ("job-1", "kb_s3_ready_queue", 1, "worker-1"))
        self.assertFalse(fail_job.call_args.args[5])
        self.assertEqual(fail_job.call_args.args[7], "s3_ready")

    def test_s3_ready_transient_failure_marks_retryable_and_archives_current_message(self) -> None:
        config = worker_config()
        with (
            patch("src.kb_parse_worker.worker.LeaseMaintainer", NoopLease),
            patch("src.kb_parse_worker.worker.claim_queue_message", return_value=claimed("s3_ready")),
            patch(
                "src.kb_parse_worker.worker.load_s3_ready_snapshot_with_short_connection",
                side_effect=RuntimeError("S3_NOT_READY_AFTER_TIMEOUT: sync delay"),
            ),
            patch(
                "src.kb_parse_worker.worker.fail_job_and_archive_current_message",
            ) as fail_job,
            patch("src.kb_parse_worker.worker.LOGGER.exception"),
        ):
            S3ReadyWorker(config).process_message(message())

        fail_job.assert_called_once()
        self.assertTrue(fail_job.call_args.args[5])
        self.assertEqual(fail_job.call_args.args[7], "s3_ready")

    def test_parse_backoff_not_due_leaves_current_message_when_contract_says_keep(self) -> None:
        conn = object()
        with patch("src.kb_parse_worker.worker.queue.archive_job_message_by_id", return_value=True) as archive:
            handle_unclaimed_message(
                conn,
                "kb_parse_queue",
                message(),
                claim_disposition("backoff_not_due", False),
            )

        archive.assert_not_called()

    def test_parse_backoff_not_due_archives_current_message_when_contract_says_archive(self) -> None:
        conn = object()
        with patch("src.kb_parse_worker.worker.queue.archive_job_message_by_id", return_value=True) as archive:
            handle_unclaimed_message(
                conn,
                "kb_parse_queue",
                message(),
                claim_disposition("backoff_not_due", True),
            )

        archive.assert_called_once_with(conn, "kb_parse_queue", 1)

    def test_parse_finalization_reconnects_and_retries_db_connection_errors(self) -> None:
        first_conn = FakeConnection("first")
        retry_conn = FakeConnection("retry")
        result = control_plane.S3ReadyEnqueueResult(
            parse_job_id="job-1",
            parse_job_status="succeeded",
            s3_ready_job_id="s3-job-1",
            s3_ready_msg_id=123,
            document_status="s3_sync_pending",
        )
        calls: list[FakeConnection] = []

        def complete(conn, *_args, **_kwargs):
            calls.append(conn)
            if len(calls) == 1:
                raise psycopg2.OperationalError("server closed the connection unexpectedly")
            return result

        with (
            patch(
                "src.kb_parse_worker.worker.control_plane.complete_parse_local_ready_and_enqueue_s3_check",
                side_effect=complete,
            ),
            patch(
                "src.kb_parse_worker.worker.control_plane.connect",
                side_effect=[first_conn, retry_conn],
            ) as connect,
            patch("src.kb_parse_worker.worker.archive_current_message", return_value=True) as archive,
            patch("src.kb_parse_worker.worker.time.sleep") as sleep,
            patch("src.kb_parse_worker.worker.LOGGER.warning"),
        ):
            actual = complete_parse_local_ready_and_archive_with_retry(
                worker_config(),
                "job-1",
                "00000000-0000-0000-0000-000000000001",
                1,
                "/processed/manifest.json",
                "00000000-0000-0000-0000-000000000002",
                "hash",
                7,
                {"processed": {}},
                {"manifest_s3_key": "processed/manifest.json"},
                "kb_parse_queue",
                1,
            )

        self.assertEqual(actual, result)
        self.assertEqual(calls, [first_conn, retry_conn])
        self.assertEqual(connect.call_count, 2)
        sleep.assert_called_once_with(1.0)
        archive.assert_called_once_with(retry_conn, "kb_parse_queue", 1)

    def test_parse_finalization_does_not_retry_non_db_errors(self) -> None:
        first_conn = FakeConnection("first")

        with (
            patch(
                "src.kb_parse_worker.worker.control_plane.complete_parse_local_ready_and_enqueue_s3_check",
                side_effect=RuntimeError(
                    "complete_parse_local_ready_and_enqueue_s3_check returned no row"
                ),
            ) as complete,
            patch("src.kb_parse_worker.worker.control_plane.connect", return_value=first_conn) as connect,
            patch("src.kb_parse_worker.worker.archive_current_message") as archive,
            patch("src.kb_parse_worker.worker.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "returned no row"):
                complete_parse_local_ready_and_archive_with_retry(
                    worker_config(),
                    "job-1",
                    "00000000-0000-0000-0000-000000000001",
                    1,
                    "/processed/manifest.json",
                    "00000000-0000-0000-0000-000000000002",
                    "hash",
                    7,
                    {"processed": {}},
                    {"manifest_s3_key": "processed/manifest.json"},
                    "kb_parse_queue",
                    1,
                )

        complete.assert_called_once()
        connect.assert_called_once_with("postgresql://test")
        archive.assert_not_called()
        sleep.assert_not_called()

    def test_s3_ready_finalization_reconnects_and_retries_db_connection_errors(self) -> None:
        first_conn = FakeConnection("first")
        retry_conn = FakeConnection("retry")
        calls: list[FakeConnection] = []

        def complete(conn, *_args, **_kwargs):
            calls.append(conn)
            if len(calls) == 1:
                raise psycopg2.InterfaceError("connection already closed")
            return True

        with (
            patch(
                "src.kb_parse_worker.worker.control_plane.complete_s3_ready_check",
                side_effect=complete,
            ),
            patch(
                "src.kb_parse_worker.worker.control_plane.connect",
                side_effect=[first_conn, retry_conn],
            ) as connect,
            patch("src.kb_parse_worker.worker.archive_current_message", return_value=True) as archive,
            patch("src.kb_parse_worker.worker.time.sleep") as sleep,
            patch("src.kb_parse_worker.worker.LOGGER.warning"),
        ):
            ok = complete_s3_ready_check_and_archive_with_retry(
                worker_config(),
                "job-1",
                "00000000-0000-0000-0000-000000000001",
                1,
                "processed/manifest.json",
                "hash",
                "00000000-0000-0000-0000-000000000002",
                "/processed/manifest.json",
                7,
                "kb_s3_ready_queue",
                1,
            )

        self.assertTrue(ok)
        self.assertEqual(calls, [first_conn, retry_conn])
        self.assertEqual(connect.call_count, 2)
        sleep.assert_called_once_with(1.0)
        archive.assert_called_once_with(retry_conn, "kb_s3_ready_queue", 1)

    def test_fail_job_reconnects_and_retries_db_connection_errors(self) -> None:
        first_conn = FakeConnection("first")
        retry_conn = FakeConnection("retry")
        result = control_plane.FailJobResult(
            job_id="job-1",
            job_status="failed",
            document_status="parse_queued",
            next_retry_at="2026-05-15T10:00:00+00:00",
            retry_wakeup_msg_id=10,
        )
        calls: list[FakeConnection] = []

        def fail(conn, *_args, **_kwargs):
            calls.append(conn)
            if len(calls) == 1:
                raise psycopg2.OperationalError("server closed the connection unexpectedly")
            return result

        with (
            patch("src.kb_parse_worker.worker.control_plane.fail_job", side_effect=fail),
            patch(
                "src.kb_parse_worker.worker.control_plane.connect",
                side_effect=[first_conn, retry_conn],
            ) as connect,
            patch("src.kb_parse_worker.worker.archive_current_message", return_value=True) as archive,
            patch("src.kb_parse_worker.worker.time.sleep") as sleep,
            patch("src.kb_parse_worker.worker.LOGGER.warning"),
        ):
            actual = fail_job_and_archive_current_message(
                worker_config(),
                "job-1",
                "kb_parse_queue",
                1,
                "worker-1",
                True,
                "temporary failure",
                "parse",
            )

        self.assertEqual(actual, result)
        self.assertEqual(calls, [first_conn, retry_conn])
        self.assertEqual(connect.call_count, 2)
        sleep.assert_called_once_with(1.0)
        archive.assert_called_once_with(retry_conn, "kb_parse_queue", 1)


if __name__ == "__main__":
    unittest.main()
