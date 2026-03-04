from pathlib import Path

from vec_kg_rag.pipeline import build_index, ingest_markdown


def test_ingest_and_build_index_offline(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    workspace = tmp_path / "workspace"
    config_dir = tmp_path / "config"

    input_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    (input_dir / "sample.md").write_text(
        "# 学院实验室与研究平台\n智能信息处理实验室。\n",
        encoding="utf-8",
    )

    ingest = ingest_markdown(
        input_dir=input_dir,
        workspace=workspace,
        processed_dir=processed_dir,
        config_dir=config_dir,
    )
    assert ingest["chunks"] > 0

    result = build_index(
        workspace=workspace,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        config_dir=config_dir,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-large",
        edge_conf_threshold=0.65,
    )
    assert result["chunks"] > 0
    assert (artifacts_dir / "kg_nodes.jsonl").exists()
    assert (workspace / "graph_index.json").exists()
