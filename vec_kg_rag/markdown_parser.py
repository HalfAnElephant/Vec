from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Chunk
from .normalize import estimate_tokens, normalize_text, normalize_title

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            lower = start + int(max_chars * 0.6)
            cut = text.rfind("\n", lower, end)
            if cut == -1:
                cut = text.rfind("。", lower, end)
            if cut == -1:
                cut = end
        else:
            cut = end

        piece = text[start:cut].strip()
        if piece:
            parts.append(piece)

        if cut >= len(text):
            break
        start = max(0, cut - overlap)

    return parts


def _chunk_id(source_file: str, section_path: str, idx: int, text: str) -> str:
    seed = f"{source_file}|{section_path}|{idx}|{text}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()[:16]


def parse_markdown_file(
    path: Path,
    source_file: str,
    alias_map: dict[str, str],
    max_chunk_chars: int = 900,
) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    headers: dict[int, str] = {}
    current_section = "ROOT"
    buffer: list[str] = []
    chunk_rows: list[Chunk] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            return

        pieces = _split_text(text, max_chars=max_chunk_chars)
        for idx, piece in enumerate(pieces):
            cid = _chunk_id(source_file, current_section, idx, piece)
            chunk_rows.append(
                Chunk(
                    chunk_id=cid,
                    text=piece,
                    source_file=source_file,
                    section_path=current_section,
                    tokens=estimate_tokens(piece),
                    node_ids=[],
                )
            )
        buffer = []

    for line in lines:
        m = _HEADING_RE.match(line.strip())
        if not m:
            buffer.append(line)
            continue

        flush()
        level = len(m.group(1))
        title = normalize_title(m.group(2).strip(), alias_map)

        drop_levels = [k for k in headers if k >= level]
        for k in drop_levels:
            headers.pop(k)
        headers[level] = title

        ordered = [headers[k] for k in sorted(headers)]
        current_section = "/".join(ordered) if ordered else "ROOT"

    flush()
    if not chunk_rows:
        # Empty markdown still gets one minimal chunk for traceability.
        text = normalize_text(raw)
        cid = _chunk_id(source_file, "ROOT", 0, text)
        chunk_rows.append(
            Chunk(
                chunk_id=cid,
                text=text,
                source_file=source_file,
                section_path="ROOT",
                tokens=estimate_tokens(text),
                node_ids=[],
            )
        )
    return chunk_rows


def collect_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.md"))
