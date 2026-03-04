from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any

from .io_utils import read_jsonl, write_json
from .query_engine import QueryEngine, has_any_keyword



def evaluate(
    workspace: Path,
    artifacts_dir: Path,
    eval_set_path: Path,
    llm_model: str,
    embedding_model: str,
    top_k: int = 5,
) -> dict[str, Any]:
    questions = read_jsonl(eval_set_path)
    if not questions:
        raise RuntimeError(f"No evaluation questions found in {eval_set_path}")

    engine = QueryEngine(
        workspace=workspace,
        artifacts_dir=artifacts_dir,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )

    details: list[dict[str, Any]] = []
    hit_count = 0
    citation_count = 0
    contain_count = 0

    for i, item in enumerate(questions, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        answer, debug = engine.answer(question, top_k=top_k)
        retrieved_texts = [c["text"] for c in debug.get("context_chunks", [])]
        retrieved_citations = [c["citation"] for c in debug.get("context_chunks", [])]

        gold_keywords = [x for x in item.get("gold_keywords", []) if isinstance(x, str)]
        expected_sources = [x for x in item.get("expected_sources", []) if isinstance(x, str)]
        expected_contains = [x for x in item.get("expected_answer_contains", []) if isinstance(x, str)]

        if gold_keywords:
            top_hit = has_any_keyword(retrieved_texts, gold_keywords)
        elif expected_sources:
            top_hit = any(src in retrieved_citations for src in expected_sources)
        else:
            top_hit = bool(answer.retrieved_chunks)

        citations_ok = bool(answer.citations)
        contains_ok = True
        if expected_contains:
            contains_ok = all(term in answer.answer for term in expected_contains)

        hit_count += int(top_hit)
        citation_count += int(citations_ok)
        contain_count += int(contains_ok)

        details.append(
            {
                "index": i,
                "question": question,
                "top5_hit": top_hit,
                "citations_ok": citations_ok,
                "contains_ok": contains_ok,
                "answer": answer.answer,
                "citations": answer.citations,
                "retrieved_chunks": answer.retrieved_chunks,
                "token_usage": debug.get("token_usage", {}),
            }
        )

    total = len(details)
    report = {
        "evaluated": total,
        "top5_hit_rate": (hit_count / total) if total else 0.0,
        "citation_coverage": (citation_count / total) if total else 0.0,
        "answer_contains_rate": (contain_count / total) if total else 0.0,
        "generated_at": int(time()),
        "details": details,
    }
    write_json(artifacts_dir / "eval_report.json", report)
    return report
