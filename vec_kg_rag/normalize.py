from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .io_utils import read_json

_DEFAULT_ALIASES = {
    "学院导师": "导师",
    "教师": "导师",
    "助理教授": "讲师",
    "副研究员": "研究员",
    "助教及其他": "助教",
}


def load_alias_map(config_dir: Path) -> dict[str, str]:
    path = config_dir / "aliases.json"
    if not path.exists():
        return dict(_DEFAULT_ALIASES)
    raw = read_json(path)
    out = dict(_DEFAULT_ALIASES)
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.strip()] = v.strip()
    return out


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_title(title: str, alias_map: dict[str, str]) -> str:
    title = normalize_text(title)
    return alias_map.get(title, title)


def normalize_name(name: str, alias_map: dict[str, str]) -> str:
    name = normalize_text(name)
    return alias_map.get(name, name)


def estimate_tokens(text: str) -> int:
    # Rough estimate usable for budgeting in mixed Chinese/English text.
    zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_zh = len(text) - zh_chars
    approx = int(zh_chars * 1.2 + non_zh * 0.3)
    return max(1, approx)
