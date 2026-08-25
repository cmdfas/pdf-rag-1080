from __future__ import annotations

import re
from typing import Any

import numpy as np

from .config import EMBED_TOP_K, HYBRID_K, RETRIEVE_K
from .index import load_bm25, load_chunks, load_embeddings, tokenize

_QUERY_ALIASES = (
    ("生存周期", "生命周期"),
    ("生存期", "生命周期"),
    ("N－S", "N-S"),
    ("NS图", "N-S图"),
)


_FILLER = re.compile(
    r"(请根据教材|请结合教材|根据教材|结合教材|请给出|给出定义|的定义|作答)"
)


def prepare_query(query: str) -> str:
    cleaned = _FILLER.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip() or query.strip()
    extra: list[str] = []
    for left, right in _QUERY_ALIASES:
        if left in cleaned and right not in cleaned:
            extra.append(right)
        elif right in cleaned and left not in cleaned:
            extra.append(left)
    if extra:
        return cleaned + " " + " ".join(extra)
    return cleaned


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def bm25_search(query: str, k: int = RETRIEVE_K) -> list[tuple[int, float]]:
    bm25, _ = load_bm25()
    tokens = tokenize(prepare_query(query))
    if not tokens:
        return []
    scores = np.array(bm25.get_scores(tokens), dtype=np.float64)
    if scores.size == 0:
        return []
    top = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def dense_search(query_vec: np.ndarray, k: int = EMBED_TOP_K) -> list[tuple[int, float]]:
    matrix = load_embeddings()
    if matrix is None or query_vec is None:
        return []
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    sims = (matrix / norms) @ q
    top = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]


def hybrid_search(
    query: str,
    query_vec: np.ndarray | None = None,
    k: int = HYBRID_K,
) -> list[dict[str, Any]]:
    chunks = load_chunks()
    lexical = bm25_search(query, k=RETRIEVE_K)
    dense = dense_search(query_vec, k=EMBED_TOP_K) if query_vec is not None else []

    fused: dict[int, float] = {}
    if lexical:
        lex_scores = np.array([s for _, s in lexical])
        lex_norm = _minmax(lex_scores)
        for (idx, _), n in zip(lexical, lex_norm):
            fused[idx] = fused.get(idx, 0.0) + 0.65 * float(n)
    if dense:
        den_scores = np.array([s for _, s in dense])
        den_norm = _minmax(den_scores)
        for (idx, _), n in zip(dense, den_norm):
            fused[idx] = fused.get(idx, 0.0) + 0.35 * float(n)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: max(k * 2, k)]
    hits = []
    for idx, score in ranked:
        item = dict(chunks[idx])
        item["score"] = float(score)
        hits.append(item)
    return _rerank(prepare_query(query), hits)[:k]


def _rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    core = re.sub(r"[什么是请根据教材给出定义、。？?（）()]", "", query)
    core = core.strip()[:12]
    for hit in hits:
        text = hit["text"]
        compact = text.replace(" ", "").replace("\n", "")
        bonus = 0.0
        if core and f"{core}是" in compact:
            bonus += 0.35
        if re.search(r"是.{4,40}(一门学科|的过程|的模型|的集合|的方法)", compact):
            bonus += 0.2
        if "简答题" in text or "学习目标" in text or "考核知识点" in text:
            bonus -= 0.18
        hit["score"] = round(hit["score"] + bonus, 4)
    return sorted(hits, key=lambda h: h["score"], reverse=True)
