from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import read_jsonl, write_jsonl
from .models import Chunk


@dataclass(slots=True)
class VectorIndex:
    embeddings: np.ndarray
    chunks: list[Chunk]

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.embeddings.size == 0 or not self.chunks:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        emb = self.embeddings
        emb_norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_norm[emb_norm == 0] = 1.0
        normalized = emb / emb_norm

        scores = normalized @ q
        if len(scores) <= top_k:
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_k)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in idx]



def save_vector_index(workspace: Path, vectors: list[list[float]], chunks: list[Chunk]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(vectors, dtype=np.float32)
    np.save(workspace / "embeddings.npy", arr)
    write_jsonl(workspace / "chunk_meta.jsonl", [c.to_dict() for c in chunks])


def load_vector_index(workspace: Path) -> VectorIndex:
    emb_path = workspace / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Vector index not found: {emb_path}")
    embeddings = np.load(emb_path)
    rows = read_jsonl(workspace / "chunk_meta.jsonl")
    chunks = [Chunk(**row) for row in rows]
    return VectorIndex(embeddings=embeddings, chunks=chunks)
