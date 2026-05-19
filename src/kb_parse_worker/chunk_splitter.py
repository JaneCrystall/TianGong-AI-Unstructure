"""Token-aware chunk splitting for KB embedding inputs."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any


class TokenCodec:
    def __init__(self) -> None:
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return len(text)

    def split(self, text: str, max_tokens: int) -> list[str]:
        if self.count(text) <= max_tokens:
            return [text]
        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            return [
                self._encoding.decode(tokens[offset : offset + max_tokens])
                for offset in range(0, len(tokens), max_tokens)
            ]
        return [text[offset : offset + max_tokens] for offset in range(0, len(text), max_tokens)]


_SENTENCE_RE = re.compile(r"[^。！？!?；;.\n]+[。！？!?；;.]*|\n+")
_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_OPEN_RE = re.compile(r"<table\b[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class SplitStats:
    source_chunk_count: int
    output_chunk_count: int
    split_parent_count: int


def _chunk_text(item: Any) -> str:
    if isinstance(item, dict):
        text = item.get("text", "")
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        return json.dumps(text, ensure_ascii=False, sort_keys=True)
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def _sentence_units(text: str) -> list[str]:
    units = [match.group(0) for match in _SENTENCE_RE.finditer(text) if match.group(0)]
    return units or [text]


def _pack_units(units: list[str], codec: TokenCodec, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        if codec.count(unit) > max_tokens:
            if current.strip():
                chunks.append(current)
                current = ""
            chunks.extend(codec.split(unit, max_tokens))
            continue
        candidate = f"{current}{unit}" if current else unit
        if current and codec.count(candidate) > max_tokens:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def _looks_like_html_table(text: str) -> bool:
    stripped = text.lstrip().lower()
    return stripped.startswith("<table") or ("<tr" in stripped and "</tr>" in stripped)


def _table_wrapper(text: str) -> tuple[str, str]:
    match = _TABLE_OPEN_RE.search(text)
    if match:
        return match.group(0), "</table>"
    return "<table>", "</table>"


def _split_table(text: str, codec: TokenCodec, max_tokens: int) -> list[str]:
    rows = [match.group(0) for match in _TR_RE.finditer(text)]
    if not rows:
        return _pack_units(_sentence_units(text), codec, max_tokens)

    open_tag, close_tag = _table_wrapper(text)
    chunks: list[str] = []
    current_rows: list[str] = []

    def render(render_rows: list[str]) -> str:
        return f"{open_tag}{''.join(render_rows)}{close_tag}"

    for row in rows:
        row_chunk = render([row])
        if codec.count(row_chunk) > max_tokens:
            if current_rows:
                chunks.append(render(current_rows))
                current_rows = []
            chunks.extend(codec.split(row_chunk, max_tokens))
            continue
        candidate_rows = [*current_rows, row]
        if current_rows and codec.count(render(candidate_rows)) > max_tokens:
            chunks.append(render(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate_rows

    if current_rows:
        chunks.append(render(current_rows))
    return [chunk for chunk in chunks if chunk.strip()]


def _split_text(text: str, codec: TokenCodec, max_tokens: int) -> list[str]:
    if codec.count(text) <= max_tokens:
        return [text]
    if _looks_like_html_table(text):
        chunks = _split_table(text, codec, max_tokens)
    else:
        chunks = _pack_units(_sentence_units(text), codec, max_tokens)
    final_chunks: list[str] = []
    for chunk in chunks:
        final_chunks.extend(codec.split(chunk, max_tokens))
    return [chunk for chunk in final_chunks if chunk.strip()]


def split_chunks_for_embedding(
    chunks: list[Any],
    document_id: str,
    max_tokens: int,
    codec: TokenCodec | None = None,
) -> tuple[list[dict[str, Any]], SplitStats]:
    if max_tokens <= 1:
        raise ValueError("max_tokens must be greater than 1")
    split_limit = max_tokens - 1
    codec = codec or TokenCodec()
    output: list[dict[str, Any]] = []
    split_parent_count = 0

    for parent_index, item in enumerate(chunks):
        base = copy.deepcopy(item) if isinstance(item, dict) else {"text": _chunk_text(item)}
        text = _chunk_text(base)
        split_texts = _split_text(text, codec, split_limit)
        if len(split_texts) > 1:
            split_parent_count += 1
        for split_text in split_texts:
            if codec.count(split_text) >= max_tokens:
                raise RuntimeError("CHUNK_SPLIT_VALIDATE_FAILED")
            child = dict(base)
            child["text"] = split_text
            output.append(child)

    return output, SplitStats(
        source_chunk_count=len(chunks),
        output_chunk_count=len(output),
        split_parent_count=split_parent_count,
    )
