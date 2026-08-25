from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import BOOK_LINK, DATA_DIR, DEFAULT_PDF, INDEX_DIR
from .extract import extract_outline, extract_pdf
from .index import build_index, load_chunks, save_embeddings
from .llm import embed_texts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_pdf(src: Path) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if BOOK_LINK.exists() or BOOK_LINK.is_symlink():
        BOOK_LINK.unlink()
    os.symlink(src.resolve(), BOOK_LINK)
    return BOOK_LINK


def ingest(pdf_path: str, with_embeddings: bool = True) -> dict:
    src = Path(pdf_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"找不到 PDF：{src}")
    _link_pdf(src)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在 OCR 抽取：{src.name}", flush=True)

    def progress(i: int, total: int) -> None:
        if i == 1 or i == total or i % 10 == 0:
            print(f"  页 {i}/{total}", flush=True)

    pages = extract_pdf(str(src), progress=progress)
    outline = extract_outline(str(src))
    meta = {
        "title": "软件工程（2024年版）",
        "authors": "张琼声",
        "source_pdf": str(src),
        "page_count": len(pages),
        "sha256": _sha256(src),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "outline": outline,
    }
    chunks = build_index(pages, meta)
    print(f"已切 {len(chunks)} 个片段，共 {len(pages)} 页。", flush=True)

    if with_embeddings:
        try:
            embed_chunks(chunks)
        except Exception as exc:
            print(f"嵌入跳过（将仅使用关键词检索）：{exc}", flush=True)
    return {"pages": len(pages), "chunks": len(chunks), "pdf": str(src)}


def embed_chunks(chunks: list | None = None) -> int:
    from .config import EMBED_PATH
    from .index import load_embeddings

    if chunks is None:
        chunks = load_chunks()
    if not chunks:
        raise RuntimeError("没有可嵌入的片段，请先 ingest。")
    texts = [f"{c.get('heading') or ''}\n{c['text']}" for c in chunks]
    existing = load_embeddings()
    start = 0
    done: list[list[float]] = []
    if existing is not None and 0 < existing.shape[0] < len(texts):
        start = int(existing.shape[0])
        done = existing.tolist()
        print(f"从第 {start + 1} 条继续（已有 {start}/{len(texts)}）", flush=True)
    print(f"正在向量化 {len(texts) - start} 个片段…", flush=True)

    def persist(batch_vectors: list) -> None:
        matrix = np.array(done + batch_vectors, dtype=np.float32)
        save_embeddings(matrix)

    rest = embed_texts(texts[start:], on_batch=persist) if start < len(texts) else []
    matrix = np.array(done + rest, dtype=np.float32)
    save_embeddings(matrix)
    print(f"嵌入完成：{matrix.shape[0]} × {matrix.shape[1]}", flush=True)
    return int(matrix.shape[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抽取教材 PDF 并建立 RAG 索引")
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_PDF), help="PDF 路径")
    parser.add_argument("--no-embed", action="store_true", help="跳过向量嵌入")
    parser.add_argument("--rebuild", action="store_true", help="用已有 OCR 结果重建索引，不重新扫描")
    parser.add_argument("--embed-only", action="store_true", help="只给已有片段做向量，不重新 OCR")
    args = parser.parse_args(argv)
    if args.embed_only:
        embed_chunks()
        return 0
    if args.rebuild:
        from .extract import repair_printed_pages
        from .index import load_meta, load_pages

        pages = repair_printed_pages(load_pages())
        meta = load_meta()
        chunks = build_index(pages, meta)
        print(f"已重建 {len(chunks)} 个片段。", flush=True)
        if not args.no_embed:
            try:
                embed_chunks(chunks)
            except Exception as exc:
                print(f"嵌入跳过（将仅使用关键词检索）：{exc}", flush=True)
        return 0
    ingest(args.pdf, with_embeddings=not args.no_embed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
