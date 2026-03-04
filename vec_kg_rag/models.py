from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Node:
    id: str
    type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    source_file: str = ""
    section_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Edge:
    src_id: str
    dst_id: str
    rel_type: str
    evidence: str
    confidence: float
    source_file: str = ""
    section_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    section_path: str
    tokens: int
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    answer: str
    citations: list[str]
    retrieved_nodes: list[str]
    retrieved_chunks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
