from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Any

from .graph_store import build_graph_index, save_graph_index
from .io_utils import append_jsonl, ensure_dir, read_jsonl, write_json, write_jsonl
from .llm_client import TokenUsage, build_model_client
from .markdown_parser import collect_markdown_files, parse_markdown_file
from .models import Chunk, Edge, Node
from .normalize import load_alias_map, normalize_name
from .ontology import normalize_entity_type, normalize_relation_type
from .vector_store import save_vector_index


def _stable_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def ingest_markdown(
    input_dir: Path,
    workspace: Path,
    processed_dir: Path,
    config_dir: Path,
) -> dict[str, Any]:
    ensure_dir(workspace)
    ensure_dir(processed_dir)

    alias_map = load_alias_map(config_dir)
    md_files = collect_markdown_files(input_dir)

    chunks: list[Chunk] = []
    file_stats: list[dict[str, Any]] = []

    for md in md_files:
        rel = md.relative_to(input_dir).as_posix()
        parsed = parse_markdown_file(md, source_file=rel, alias_map=alias_map)
        chunks.extend(parsed)
        file_stats.append({"source_file": rel, "chunks": len(parsed)})

    write_jsonl(processed_dir / "chunks.jsonl", [c.to_dict() for c in chunks])
    write_jsonl(processed_dir / "documents.jsonl", file_stats)
    write_json(workspace / "ingest_stats.json", {"files": len(md_files), "chunks": len(chunks)})

    return {"files": len(md_files), "chunks": len(chunks), "processed_path": str(processed_dir / "chunks.jsonl")}


def _coerce_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = payload.get("nodes", [])
    return nodes if isinstance(nodes, list) else []


def _coerce_edges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = payload.get("edges", [])
    return edges if isinstance(edges, list) else []


def build_index(
    workspace: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    config_dir: Path,
    llm_model: str,
    embedding_model: str,
    edge_conf_threshold: float = 0.65,
) -> dict[str, Any]:
    ensure_dir(workspace)
    ensure_dir(artifacts_dir)

    chunk_rows = read_jsonl(processed_dir / "chunks.jsonl")
    chunks = [Chunk(**row) for row in chunk_rows]
    if not chunks:
        raise RuntimeError("No processed chunks found. Run ingest first.")

    alias_map = load_alias_map(config_dir)
    model = build_model_client(llm_model=llm_model, embedding_model=embedding_model)

    node_by_key: dict[str, Node] = {}
    name_to_ids: dict[str, list[str]] = {}
    edge_seen: set[str] = set()
    edges_main: list[Edge] = []
    edges_candidates: list[Edge] = []
    total_usage = TokenUsage()

    def upsert_node(name: str, ntype: str, summary: str, source_file: str, section_path: str, aliases: list[str]) -> str:
        clean_name = normalize_name(name, alias_map)
        clean_type = normalize_entity_type(ntype)
        key = f"{clean_type}|{clean_name}|{section_path}"
        if key not in node_by_key:
            node = Node(
                id=_stable_id(key),
                type=clean_type,
                name=clean_name,
                aliases=sorted({normalize_name(a, alias_map) for a in aliases if a}),
                summary=(summary or "")[:300],
                source_file=source_file,
                section_path=section_path,
            )
            node_by_key[key] = node
            name_to_ids.setdefault(clean_name, []).append(node.id)
        else:
            existing = node_by_key[key]
            merged_aliases = set(existing.aliases)
            merged_aliases.update(normalize_name(a, alias_map) for a in aliases if a)
            existing.aliases = sorted(merged_aliases)
            if not existing.summary and summary:
                existing.summary = summary[:300]
        return node_by_key[key].id

    for chunk in chunks:
        payload, usage = model.extract_kg(chunk.text, chunk.source_file, chunk.section_path)
        total_usage.add(usage)

        local_name_to_id: dict[str, str] = {}
        node_ids_for_chunk: set[str] = set()

        for raw_node in _coerce_nodes(payload):
            if not isinstance(raw_node, dict):
                continue
            name = str(raw_node.get("name", "")).strip()
            if not name:
                continue
            ntype = str(raw_node.get("type", "Resource"))
            summary = str(raw_node.get("summary", ""))
            aliases = raw_node.get("aliases", [])
            aliases = aliases if isinstance(aliases, list) else []
            nid = upsert_node(
                name=name,
                ntype=ntype,
                summary=summary,
                source_file=chunk.source_file,
                section_path=chunk.section_path,
                aliases=[str(x) for x in aliases],
            )
            canonical_name = normalize_name(name, alias_map)
            local_name_to_id[canonical_name] = nid
            node_ids_for_chunk.add(nid)

        for raw_edge in _coerce_edges(payload):
            if not isinstance(raw_edge, dict):
                continue
            src_name = normalize_name(str(raw_edge.get("src", "")).strip(), alias_map)
            dst_name = normalize_name(str(raw_edge.get("dst", "")).strip(), alias_map)
            if not src_name or not dst_name:
                continue

            src_id = local_name_to_id.get(src_name)
            dst_id = local_name_to_id.get(dst_name)

            if not src_id and len(name_to_ids.get(src_name, [])) == 1:
                src_id = name_to_ids[src_name][0]
            if not dst_id and len(name_to_ids.get(dst_name, [])) == 1:
                dst_id = name_to_ids[dst_name][0]

            if not src_id:
                src_id = upsert_node(
                    name=src_name,
                    ntype="Resource",
                    summary="由关系抽取补全",
                    source_file=chunk.source_file,
                    section_path=chunk.section_path,
                    aliases=[],
                )
            if not dst_id:
                dst_id = upsert_node(
                    name=dst_name,
                    ntype="Resource",
                    summary="由关系抽取补全",
                    source_file=chunk.source_file,
                    section_path=chunk.section_path,
                    aliases=[],
                )

            node_ids_for_chunk.update({src_id, dst_id})
            rel_type = normalize_relation_type(str(raw_edge.get("rel_type", "")))
            evidence = str(raw_edge.get("evidence", "")).strip()[:400]
            try:
                confidence = float(raw_edge.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            edge_key = f"{src_id}|{dst_id}|{rel_type}|{chunk.section_path}|{evidence}"
            if edge_key in edge_seen:
                continue
            edge_seen.add(edge_key)

            edge = Edge(
                src_id=src_id,
                dst_id=dst_id,
                rel_type=rel_type,
                evidence=evidence,
                confidence=confidence,
                source_file=chunk.source_file,
                section_path=chunk.section_path,
            )
            if confidence >= edge_conf_threshold:
                edges_main.append(edge)
            else:
                edges_candidates.append(edge)

        chunk.node_ids = sorted(node_ids_for_chunk)

    all_nodes = list(node_by_key.values())

    try:
        embeddings, emb_usage = model.embed_texts([c.text for c in chunks])
    except RuntimeError as e:
        raise RuntimeError(
            f"Embedding failed: {e}. "
            "If you use Ollama embeddings, run `ollama pull qwen3-embedding:0.6b` first."
        ) from e
    total_usage.add(emb_usage)
    save_vector_index(workspace, embeddings, chunks)

    graph = build_graph_index(all_nodes, edges_main)
    save_graph_index(workspace / "graph_index.json", graph)

    write_jsonl(processed_dir / "chunks_enriched.jsonl", [c.to_dict() for c in chunks])
    write_jsonl(artifacts_dir / "kg_nodes.jsonl", [n.to_dict() for n in all_nodes])
    write_jsonl(artifacts_dir / "kg_edges.jsonl", [e.to_dict() for e in edges_main])
    write_jsonl(artifacts_dir / "kg_edges_candidates.jsonl", [e.to_dict() for e in edges_candidates])

    write_json(
        workspace / "index_stats.json",
        {
            "nodes": len(all_nodes),
            "edges_main": len(edges_main),
            "edges_candidates": len(edges_candidates),
            "chunks": len(chunks),
            "token_usage": total_usage.to_dict(),
            "edge_conf_threshold": edge_conf_threshold,
            "built_at": int(time()),
        },
    )

    append_jsonl(
        artifacts_dir / "query_logs.jsonl",
        {
            "type": "build-index",
            "timestamp": int(time()),
            "token_usage": total_usage.to_dict(),
            "nodes": len(all_nodes),
            "edges_main": len(edges_main),
            "edges_candidates": len(edges_candidates),
            "chunks": len(chunks),
        },
    )

    return {
        "nodes": len(all_nodes),
        "edges_main": len(edges_main),
        "edges_candidates": len(edges_candidates),
        "chunks": len(chunks),
        "token_usage": total_usage.to_dict(),
    }
