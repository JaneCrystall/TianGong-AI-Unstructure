"""HTTP adapter for Unstructure-Serve parser APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class ParserError(RuntimeError):
    pass


class ParserTimeout(ParserError):
    pass


class ParserTaskFailure(ParserError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    result: list[Any]
    txt: str | None
    original_chunk_count: int
    dropped_empty_text_count: int


def _has_nonempty_text(item: Any) -> bool:
    if isinstance(item, dict):
        text = item.get("text")
        return isinstance(text, str) and bool(text.strip())
    if isinstance(item, str):
        return bool(item.strip())
    return True


def filter_empty_text_chunks(result: list[Any]) -> tuple[list[Any], int]:
    filtered = [item for item in result if _has_nonempty_text(item)]
    return filtered, len(result) - len(filtered)


def infer_two_stage_base_url(api_url: str) -> str:
    trimmed = api_url.rstrip("/")
    for suffix in ("/mineru_with_images", "/mineru", "/two_stage/task"):
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
            break
    return trimmed.rstrip("/")


def _parsed_document_from_payload(payload: dict[str, Any], return_txt: bool) -> ParsedDocument:
    if "result" not in payload:
        raise ParserError("parser response missing result")
    result = payload["result"]
    if not isinstance(result, list):
        raise ParserError("parser result must be a list")
    filtered_result, dropped_count = filter_empty_text_chunks(result)
    txt = payload.get("txt")
    if txt is not None and not isinstance(txt, str):
        raise ParserError("parser txt must be a string when present")
    if return_txt and txt is None:
        raise ParserError("parser response missing txt")
    return ParsedDocument(
        result=filtered_result,
        txt=txt,
        original_chunk_count=len(result),
        dropped_empty_text_count=dropped_count,
    )


def _raise_for_parser_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise ParserError(f"parser http error {response.status_code}: {response.text[:500]}") from exc


def _parse_with_two_stage(
    raw_path: Path,
    base_url: str,
    bearer_token: str,
    *,
    timeout_seconds: int,
    return_txt: bool,
    submit_timeout_seconds: int,
    status_timeout_seconds: int,
    poll_interval_seconds: int,
    priority: str,
    chunk_type: bool,
    provider: str | None,
    model: str | None,
    prompt: str | None,
) -> ParsedDocument:
    if timeout_seconds <= 0:
        raise ParserTimeout("parser two-stage timeout before submit")
    deadline = time.monotonic() + timeout_seconds
    headers = {"Authorization": f"Bearer {bearer_token}"}
    submit_url = f"{base_url.rstrip('/')}/two_stage/task"
    status_base_url = submit_url
    form_data: dict[str, str] = {
        "return_txt": "true" if return_txt else "false",
        "chunk_type": "true" if chunk_type else "false",
        "priority": priority or "normal",
    }
    if provider:
        form_data["provider"] = provider
    if model:
        form_data["model"] = model
    if prompt:
        form_data["prompt"] = prompt

    with raw_path.open("rb") as handle:
        response = requests.post(
            submit_url,
            files={"file": (raw_path.name, handle)},
            data=form_data,
            headers=headers,
            timeout=min(submit_timeout_seconds, max(1, int(deadline - time.monotonic()))),
        )
    _raise_for_parser_status(response)
    submit_payload = response.json()
    task_id = submit_payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ParserError("parser response missing task_id")

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ParserTimeout(f"parser two-stage task {task_id} timed out")
        status_response = requests.get(
            f"{status_base_url}/{task_id}",
            headers=headers,
            timeout=min(status_timeout_seconds, max(1, int(remaining))),
        )
        _raise_for_parser_status(status_response)
        status_payload = status_response.json()
        state = status_payload.get("state")
        if state == "SUCCESS":
            result_payload = status_payload.get("result")
            if not isinstance(result_payload, dict):
                raise ParserError("parser response missing result payload")
            return _parsed_document_from_payload(result_payload, return_txt)
        if state in {"FAILURE", "REVOKED"}:
            error_detail = status_payload.get("error") or state
            raise ParserTaskFailure(f"parser two-stage task {task_id} failed: {error_detail}")
        if not isinstance(state, str) or not state:
            raise ParserError("parser response missing state")
        time.sleep(min(poll_interval_seconds, max(0.1, deadline - time.monotonic())))


def parse_with_unstructure_serve(
    raw_path: Path,
    api_url: str,
    bearer_token: str,
    timeout_seconds: int = 3600,
    return_txt: bool = True,
    *,
    use_two_stage: bool = False,
    two_stage_base_url: str | None = None,
    two_stage_submit_timeout_seconds: int = 120,
    two_stage_status_timeout_seconds: int = 30,
    two_stage_poll_interval_seconds: int = 3,
    two_stage_priority: str = "normal",
    two_stage_chunk_type: bool = True,
    two_stage_provider: str | None = None,
    two_stage_model: str | None = None,
    two_stage_prompt: str | None = None,
) -> ParsedDocument:
    if use_two_stage:
        return _parse_with_two_stage(
            raw_path,
            two_stage_base_url or infer_two_stage_base_url(api_url),
            bearer_token,
            timeout_seconds=timeout_seconds,
            return_txt=return_txt,
            submit_timeout_seconds=two_stage_submit_timeout_seconds,
            status_timeout_seconds=two_stage_status_timeout_seconds,
            poll_interval_seconds=two_stage_poll_interval_seconds,
            priority=two_stage_priority,
            chunk_type=two_stage_chunk_type,
            provider=two_stage_provider,
            model=two_stage_model,
            prompt=two_stage_prompt,
        )

    headers = {"Authorization": f"Bearer {bearer_token}"}
    with raw_path.open("rb") as handle:
        response = requests.post(
            api_url,
            files={"file": (raw_path.name, handle)},
            params={"return_txt": "true" if return_txt else "false"},
            data={"return_txt": "true" if return_txt else "false"},
            headers=headers,
            timeout=timeout_seconds,
        )
    _raise_for_parser_status(response)

    return _parsed_document_from_payload(response.json(), return_txt)
