from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .config import (
    BM25_PATH,
    CHUNK_OVERLAP_LINES,
    CHUNK_SIZE,
    CHUNKS_PATH,
    EMBED_PATH,
    INDEX_DIR,
    META_PATH,
    PAGES_PATH,
)

_END_PUNCT = re.compile(r"[。！？；：.!?]$")
_NEW_BLOCK = re.compile(
    r"^(?:第[一二三四五六七八九十百0-9]+[章节节]|(?:[一二三四五六七八九十]+、)|(?:\d+[.)、])|(?:[（(]\d+[)）]))"
)


def join_wrapped(texts: list[str]) -> str:
    """Merge visual wraps so OCR-split words like 软件工 / 程是 become 软件工程是."""
    out: list[str] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if (
            out
            and len(out[-1]) >= 22
            and not _END_PUNCT.search(out[-1])
            and not _NEW_BLOCK.match(text)
        ):
            out[-1] += text
        else:
            out.append(text)
    return "\n".join(out)


STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "对", "中", "等",
    "其", "这", "那", "也", "就", "都", "而", "并", "以", "把", "被", "由",
    "从", "到", "于", "上", "下", "一个", "一种", "可以", "通过", "进行",
}


def tokenize(text: str) -> list[str]:
    tokens = [t.strip().lower() for t in jieba.cut_for_search(text) if t.strip()]
    return [t for t in tokens if t not in STOPWORDS and t not in "，。、；：？！“”‘’（）【】《》—…·"]


def _body_lines(page: dict[str, Any]) -> list[dict[str, Any]]:
    return [ln for ln in page["lines"] if ln["region"] == "body" and ln["text"]]


def _pack_chunks(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_heading = ""
    current_chapter = ""
    buffer: list[tuple[dict[str, Any], dict[str, Any]]] = []
    chunks: list[dict[str, Any]] = []

    def flush(force: bool = False) -> None:
        nonlocal buffer
        if not buffer:
            return
        text_len = sum(len(ln["text"]) for _, ln in buffer)
        if not force and text_len < CHUNK_SIZE:
            return
        first_page, first_line = buffer[0]
        last_page, last_line = buffer[-1]
        raw_lines = [ln["text"] for _, ln in buffer]
        text = join_wrapped(raw_lines)
        page_set = []
        seen = set()
        for pg, ln in buffer:
            key = pg["pdf_page"]
            if key not in seen:
                seen.add(key)
                page_set.append(
                    {
                        "pdf_page": pg["pdf_page"],
                        "printed_page": pg.get("printed_page"),
                        "line_start": ln["line_no"],
                        "line_end": ln["line_no"],
                    }
                )
            else:
                page_set[-1]["line_end"] = ln["line_no"]
        chunks.append(
            {
                "id": f"C{len(chunks) + 1:04d}",
                "text": text,
                "raw_lines": raw_lines,
                "heading": current_heading,
                "chapter": current_chapter,
                "pdf_page": first_page["pdf_page"],
                "printed_page": first_page.get("printed_page"),
                "pdf_page_end": last_page["pdf_page"],
                "printed_page_end": last_page.get("printed_page"),
                "line_start": first_line["line_no"],
                "line_end": last_line["line_no"],
                "pages": page_set,
                "bboxes": [
                    {
                        "pdf_page": pg["pdf_page"],
                        "line_no": ln["line_no"],
                        **ln["bbox"],
                        "page_width": ln["page_width"],
                        "page_height": ln["page_height"],
                    }
                    for pg, ln in buffer
                ],
            }
        )
        if CHUNK_OVERLAP_LINES and not force:
            buffer = buffer[-CHUNK_OVERLAP_LINES:]
        else:
            buffer = []

    for page in pages:
        for heading in page.get("headings") or []:
            current_heading = heading["text"]
            if heading["text"].startswith("第") and "章" in heading["text"][:8]:
                current_chapter = heading["text"]
        for line in _body_lines(page):
            if any(h["line_no"] == line["line_no"] for h in page.get("headings") or []):
                flush(force=True)
                current_heading = line["text"]
                if line["text"].startswith("第") and "章" in line["text"][:8]:
                    current_chapter = line["text"]
            buffer.append((page, line))
            if sum(len(ln["text"]) for _, ln in buffer) >= CHUNK_SIZE:
                flush()
    flush(force=True)
    return chunks


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_index(pages: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = _pack_chunks(pages)
    save_jsonl(PAGES_PATH, pages)
    save_jsonl(CHUNKS_PATH, chunks)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with BM25_PATH.open("wb") as fh:
        pickle.dump({"bm25": bm25, "tokens": corpus}, fh)
    return chunks


def load_chunks() -> list[dict[str, Any]]:
    return load_jsonl(CHUNKS_PATH)


def load_pages() -> list[dict[str, Any]]:
    return load_jsonl(PAGES_PATH)


def load_meta() -> dict[str, Any]:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def load_bm25() -> tuple[BM25Okapi, list[list[str]]]:
    with BM25_PATH.open("rb") as fh:
        payload = pickle.load(fh)
    return payload["bm25"], payload["tokens"]


def save_embeddings(vectors: np.ndarray) -> None:
    np.save(EMBED_PATH, vectors)


def load_embeddings() -> np.ndarray | None:
    if not EMBED_PATH.exists():
        return None
    return np.load(EMBED_PATH)


def index_ready() -> bool:
    return CHUNKS_PATH.exists() and BM25_PATH.exists() and META_PATH.exists()
