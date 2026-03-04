from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
from openai import OpenAI

from .ontology import ENTITY_TYPES, RELATION_TYPES


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _parse_usage(raw: Any) -> TokenUsage:
    if not raw:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(raw, "total_tokens", 0) or 0),
    )


def _safe_json_load(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)


class DeepSeekChatClient:
    """DeepSeek chat client (OpenAI-compatible API) for KG extraction and RAG answer."""

    def __init__(self, llm_model: str = "deepseek-chat") -> None:
        self.llm_model = llm_model
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.enabled = bool(self.api_key)
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url)
            if self.enabled
            else None
        )

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0,
    ) -> tuple[dict[str, Any], TokenUsage]:
        if not self.enabled or self.client is None:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek chat calls.")

        kwargs = {
            "model": self.llm_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # DeepSeek compatibility can vary by endpoint; retry without response_format if needed.
        try:
            resp = self.client.chat.completions.create(
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            resp = self.client.chat.completions.create(**kwargs)

        content = resp.choices[0].message.content or "{}"
        data = _safe_json_load(content)
        return data, _parse_usage(resp.usage)

    def extract_kg(
        self, chunk_text: str, source_file: str, section_path: str
    ) -> tuple[dict[str, Any], TokenUsage]:
        system_prompt = (
            "你是知识图谱抽取器。严格输出 JSON。"
            "实体类型仅允许: "
            + ", ".join(sorted(ENTITY_TYPES))
            + "。关系类型仅允许: "
            + ", ".join(sorted(RELATION_TYPES))
            + "；若无法匹配，使用 OTHER_*。"
            "只抽取文本中有明确证据的实体与关系。"
        )
        user_prompt = (
            f"source_file: {source_file}\n"
            f"section_path: {section_path}\n"
            "请输出结构:"
            '{"nodes":[{"name":"","type":"","aliases":[],"summary":""}],'
            '"edges":[{"src":"","dst":"","rel_type":"","evidence":"","confidence":0.0}]}'
            "\n文本:\n"
            f"{chunk_text}"
        )
        return self._chat_json(system_prompt, user_prompt)

    def answer(
        self,
        question: str,
        context_chunks: list[dict[str, str]],
        context_edges: list[dict[str, str]],
    ) -> tuple[dict[str, Any], TokenUsage]:
        chunks_text = "\n\n".join(
            f"[C{i+1}] {c['citation']}\n{c['text']}" for i, c in enumerate(context_chunks)
        )
        edges_text = "\n".join(
            f"- {e['src']} --{e['rel_type']}--> {e['dst']} ({e['evidence']})"
            for e in context_edges
        )
        user_prompt = (
            "你是学院知识库问答助手。基于给定证据回答，不要编造。"
            "输出 JSON: {\"answer\":\"\",\"citations\":[],\"confidence\":\"high|medium|low\"}."
            "citations 只能引用给定的 citation 标签。"
            f"\n\n问题:\n{question}\n\n文本证据:\n{chunks_text}\n\n图关系证据:\n{edges_text}"
        )
        return self._chat_json("你是严谨的 RAG 回答器。", user_prompt)

    def extract_query_entities(self, question: str) -> tuple[list[str], TokenUsage]:
        user_prompt = (
            "从问题中提取关键词实体，输出 JSON {\"entities\": [\"...\"]}。"
            "优先提取学院、系所、导师、实验室、企业等名词。"
            f"\n问题: {question}"
        )
        data, usage = self._chat_json("你是信息抽取助手。", user_prompt)
        entities = [e for e in data.get("entities", []) if isinstance(e, str) and e.strip()]
        return entities, usage


class OllamaEmbeddingClient:
    """Local Ollama embedding client for vectorization."""

    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        base_url: str | None = None,
        timeout_sec: int = 120,
    ) -> None:
        self.embedding_model = embedding_model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_sec = timeout_sec

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}: {e}") from e

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        if not texts:
            return [], TokenUsage()

        # Preferred endpoint for newer Ollama versions.
        try:
            data = self._post_json(
                "/api/embed",
                {"model": self.embedding_model, "input": texts, "truncate": True},
            )
            embeddings = data.get("embeddings", [])
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return embeddings, TokenUsage(total_tokens=sum(max(1, len(t) // 4) for t in texts))
        except RuntimeError:
            pass

        # Backward-compatible endpoint.
        vectors: list[list[float]] = []
        for text in texts:
            data = self._post_json(
                "/api/embeddings",
                {"model": self.embedding_model, "prompt": text},
            )
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(
                    "Ollama embedding response malformed. "
                    f"Please ensure model `{self.embedding_model}` exists."
                )
            vectors.append(emb)

        usage = TokenUsage(total_tokens=sum(max(1, len(t) // 4) for t in texts))
        return vectors, usage


class HybridModelClient:
    """DeepSeek for chat + Ollama for embeddings."""

    def __init__(self, chat_client: DeepSeekChatClient, embedding_client: OllamaEmbeddingClient) -> None:
        self.chat_client = chat_client
        self.embedding_client = embedding_client

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        return self.embedding_client.embed_texts(texts)

    def extract_kg(
        self, chunk_text: str, source_file: str, section_path: str
    ) -> tuple[dict[str, Any], TokenUsage]:
        return self.chat_client.extract_kg(chunk_text, source_file, section_path)

    def answer(
        self,
        question: str,
        context_chunks: list[dict[str, str]],
        context_edges: list[dict[str, str]],
    ) -> tuple[dict[str, Any], TokenUsage]:
        return self.chat_client.answer(question, context_chunks, context_edges)

    def extract_query_entities(self, question: str) -> tuple[list[str], TokenUsage]:
        return self.chat_client.extract_query_entities(question)


class HeuristicModelClient:
    """Offline fallback for local testing without API keys."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _hash_embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        terms = re.findall(r"[\w\u4e00-\u9fff]+", text)
        if not terms:
            terms = list(text)
        for term in terms:
            idx = hash(term) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        return [self._hash_embed(t) for t in texts], TokenUsage(total_tokens=sum(len(t) for t in texts) // 4)

    def extract_kg(
        self, chunk_text: str, source_file: str, section_path: str
    ) -> tuple[dict[str, Any], TokenUsage]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        if "学院" in section_path:
            head = section_path.split("/")[0]
            nodes.append(
                {
                    "name": head,
                    "type": "College",
                    "aliases": [],
                    "summary": f"来源于 {section_path} 的学院节点",
                }
            )

        if "系" in section_path or "系" in chunk_text:
            for m in re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{2,30}系)", chunk_text):
                nodes.append({"name": m, "type": "Department", "aliases": [], "summary": "系所信息"})

        for m in re.findall(r"([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:实验室|研究中心|平台))", chunk_text):
            nodes.append({"name": m, "type": "Lab", "aliases": [], "summary": "实验室或平台"})

        for m in re.findall(r"([\u4e00-\u9fff]{2,8}(?:教授|副教授|讲师|研究员))", chunk_text):
            nodes.append({"name": m, "type": "Faculty", "aliases": [], "summary": "导师或教师"})

        if "实习" in chunk_text and "企业" in chunk_text:
            edges.append(
                {
                    "src": section_path.split("/")[0] if section_path else "学院",
                    "dst": "企业合作",
                    "rel_type": "COOPERATES_WITH",
                    "evidence": "文本中出现实习与企业",
                    "confidence": 0.7,
                }
            )

        return {"nodes": nodes, "edges": edges}, TokenUsage(total_tokens=max(1, len(chunk_text) // 4))

    def answer(
        self,
        question: str,
        context_chunks: list[dict[str, str]],
        context_edges: list[dict[str, str]],
    ) -> tuple[dict[str, Any], TokenUsage]:
        citations = [item["citation"] for item in context_chunks[:3]]
        if not context_chunks:
            answer = "未检索到相关证据，建议补充文档后重试。"
            conf = "low"
        else:
            snippets = "；".join(item["text"][:80] for item in context_chunks[:2])
            answer = f"基于已检索证据，问题“{question}”相关信息包括：{snippets}。"
            conf = "medium"
        return {"answer": answer, "citations": citations, "confidence": conf}, TokenUsage(total_tokens=len(question) // 2)

    def extract_query_entities(self, question: str) -> tuple[list[str], TokenUsage]:
        entities = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", question)
        return entities[:5], TokenUsage(total_tokens=max(1, len(question) // 4))


def build_model_client(
    llm_model: str = "deepseek-chat",
    embedding_model: str = "qwen3-embedding:0.6b",
):
    mode = os.getenv("VEC_MODEL_MODE", "auto").lower()
    if mode == "offline":
        return HeuristicModelClient()

    chat_client = DeepSeekChatClient(llm_model=llm_model)
    if chat_client.enabled:
        embedding_client = OllamaEmbeddingClient(embedding_model=embedding_model)
        return HybridModelClient(chat_client=chat_client, embedding_client=embedding_client)

    return HeuristicModelClient()
