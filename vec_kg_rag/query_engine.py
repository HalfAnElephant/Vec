from __future__ import annotations

import re
from pathlib import Path
from time import time
from typing import Any

from .graph_store import load_graph_index
from .io_utils import append_jsonl
from .llm_client import TokenUsage, build_model_client
from .models import Answer
from .vector_store import load_vector_index


class QueryEngine:
    def __init__(
        self,
        workspace: Path,
        artifacts_dir: Path,
        llm_model: str,
        embedding_model: str,
    ) -> None:
        self.workspace = workspace
        self.artifacts_dir = artifacts_dir
        self.vector_index = load_vector_index(workspace)
        self.graph = load_graph_index(workspace / "graph_index.json")
        self.model = build_model_client(llm_model=llm_model, embedding_model=embedding_model)

    def retrieve(self, question: str, top_k: int = 5, graph_hops: int = 1) -> dict[str, Any]:
        usage = TokenUsage()
        try:
            q_embs, u1 = self.model.embed_texts([question])
        except RuntimeError as e:
            raise RuntimeError(
                f"Query embedding failed: {e}. "
                "If you use Ollama embeddings, run `ollama pull qwen3-embedding:0.6b` first."
            ) from e
        usage.add(u1)
        hits = self.vector_index.search(q_embs[0], top_k=top_k)

        seed_node_ids: set[str] = set()
        context_chunks: list[dict[str, str]] = []
        for chunk, score in hits:
            seed_node_ids.update(chunk.node_ids)
            context_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "citation": f"{chunk.source_file}#{chunk.section_path}",
                    "score": f"{score:.4f}",
                }
            )

        entities, u2 = self.model.extract_query_entities(question)
        usage.add(u2)

        for entity in entities:
            for node in self.graph.nodes.values():
                if entity and (entity in node.name or node.name in entity):
                    seed_node_ids.add(node.id)

        expanded_node_ids = set(self.graph.expand_neighbors(list(seed_node_ids), hops=graph_hops))
        related_nodes = [self.graph.nodes[nid] for nid in expanded_node_ids if nid in self.graph.nodes]
        related_edges = self.graph.related_edges(expanded_node_ids)

        edge_context = []
        for edge in related_edges[:30]:
            src = self.graph.nodes.get(edge.src_id)
            dst = self.graph.nodes.get(edge.dst_id)
            edge_context.append(
                {
                    "src": src.name if src else edge.src_id,
                    "dst": dst.name if dst else edge.dst_id,
                    "rel_type": edge.rel_type,
                    "evidence": edge.evidence,
                }
            )

        return {
            "context_chunks": context_chunks,
            "related_nodes": related_nodes,
            "related_edges": edge_context,
            "usage": usage,
        }

    def answer(self, question: str, top_k: int = 5, graph_hops: int = 1) -> tuple[Answer, dict[str, Any]]:
        retrieved = self.retrieve(question, top_k=top_k, graph_hops=graph_hops)
        usage = TokenUsage()
        usage.add(retrieved["usage"])

        result, u3 = self.model.answer(
            question=question,
            context_chunks=retrieved["context_chunks"],
            context_edges=retrieved["related_edges"],
        )
        usage.add(u3)

        allowed_citations = {c["citation"] for c in retrieved["context_chunks"]}
        citations_raw = result.get("citations", []) if isinstance(result, dict) else []
        citations = [c for c in citations_raw if isinstance(c, str) and c in allowed_citations]

        answer_text = ""
        if isinstance(result, dict):
            answer_text = str(result.get("answer", "")).strip()

        if not citations:
            conf = str(result.get("confidence", "")).lower() if isinstance(result, dict) else ""
            if conf != "high":
                answer_text = (answer_text + "\n\n[提示] 当前答案缺少可验证引用，可信度较低。").strip()

        answer = Answer(
            answer=answer_text,
            citations=citations,
            retrieved_nodes=[n.id for n in retrieved["related_nodes"]],
            retrieved_chunks=[c["chunk_id"] for c in retrieved["context_chunks"]],
        )

        log_row = {
            "type": "query",
            "timestamp": int(time()),
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "retrieved_nodes": answer.retrieved_nodes,
            "retrieved_chunks": answer.retrieved_chunks,
            "token_usage": usage.to_dict(),
        }
        append_jsonl(self.artifacts_dir / "query_logs.jsonl", log_row)

        debug = {
            "context_chunks": retrieved["context_chunks"],
            "related_edges": retrieved["related_edges"],
            "token_usage": usage.to_dict(),
        }
        return answer, debug


def has_any_keyword(texts: list[str], keywords: list[str]) -> bool:
    blob = "\n".join(texts)
    return any(k and re.search(re.escape(k), blob, flags=re.IGNORECASE) for k in keywords)
