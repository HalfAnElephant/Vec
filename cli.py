from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vec_kg_rag.evaluation import evaluate
from vec_kg_rag.pipeline import build_index, ingest_markdown
from vec_kg_rag.query_engine import QueryEngine


def _default_llm_model() -> str:
    return os.getenv("VEC_LLM_MODEL", "deepseek-chat")


def _default_embedding_model() -> str:
    return os.getenv("VEC_EMBEDDING_MODEL", "qwen3-embedding:0.6b")


def cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest_markdown(
        input_dir=Path(args.input),
        workspace=Path(args.workspace),
        processed_dir=Path("data/processed"),
        config_dir=Path("config"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_build_index(args: argparse.Namespace) -> None:
    result = build_index(
        workspace=Path(args.workspace),
        processed_dir=Path("data/processed"),
        artifacts_dir=Path("artifacts"),
        config_dir=Path("config"),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        edge_conf_threshold=args.edge_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    engine = QueryEngine(
        workspace=Path(args.workspace),
        artifacts_dir=Path("artifacts"),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    answer, debug = engine.answer(question=args.q, top_k=args.top_k, graph_hops=args.graph_hops)
    out = answer.to_dict()
    out["debug"] = {
        "context_chunks": debug.get("context_chunks", []),
        "token_usage": debug.get("token_usage", {}),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluate(
        workspace=Path(args.workspace),
        artifacts_dir=Path("artifacts"),
        eval_set_path=Path(args.set),
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Markdown -> KG -> RAG prototype CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse markdown files into normalized chunks")
    p_ingest.add_argument("--input", required=True, help="Input markdown root directory")
    p_ingest.add_argument("--workspace", default=".lightrag", help="Workspace path")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build-index", help="Extract KG and build vector/graph indices")
    p_build.add_argument("--workspace", default=".lightrag", help="Workspace path")
    p_build.add_argument("--llm-model", default=_default_llm_model(), help="LLM model name")
    p_build.add_argument(
        "--embedding-model",
        default=_default_embedding_model(),
        help="Embedding model name",
    )
    p_build.add_argument(
        "--edge-threshold",
        type=float,
        default=0.65,
        help="Main graph confidence threshold",
    )
    p_build.set_defaults(func=cmd_build_index)

    p_query = sub.add_parser("query", help="Run RAG query with citations")
    p_query.add_argument("--workspace", default=".lightrag", help="Workspace path")
    p_query.add_argument("--q", required=True, help="Question")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--graph-hops", type=int, default=1)
    p_query.add_argument("--llm-model", default=_default_llm_model(), help="LLM model name")
    p_query.add_argument(
        "--embedding-model",
        default=_default_embedding_model(),
        help="Embedding model name",
    )
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", help="Evaluate retrieval and answer quality")
    p_eval.add_argument("--workspace", default=".lightrag", help="Workspace path")
    p_eval.add_argument("--set", required=True, help="Evaluation set JSONL path")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--llm-model", default=_default_llm_model(), help="LLM model name")
    p_eval.add_argument(
        "--embedding-model",
        default=_default_embedding_model(),
        help="Embedding model name",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
