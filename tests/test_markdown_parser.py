from pathlib import Path

from vec_kg_rag.markdown_parser import parse_markdown_file


def test_parse_markdown_sections(tmp_path: Path) -> None:
    md = tmp_path / "a.md"
    md.write_text(
        "# 学院概况\n学院介绍。\n\n## 学院导师\n张三教授。\n",
        encoding="utf-8",
    )
    chunks = parse_markdown_file(md, source_file="a.md", alias_map={})
    assert chunks
    assert any(c.section_path == "学院概况" for c in chunks)
    assert any(c.section_path == "学院概况/学院导师" for c in chunks)
