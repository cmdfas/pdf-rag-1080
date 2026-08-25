from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdf_rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_p = sub.add_parser("ingest", help="抽取 PDF 并建立索引")
    ingest_p.add_argument("pdf", nargs="?")
    ingest_p.add_argument("--no-embed", action="store_true")
    ingest_p.add_argument("--rebuild", action="store_true")
    ingest_p.add_argument("--embed-only", action="store_true")

    ask_p = sub.add_parser("ask", help="命令行提问")
    ask_p.add_argument("question")
    ask_p.add_argument("--no-embed", action="store_true")

    serve_p = sub.add_parser("serve", help="启动网页问答")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from .config import DEFAULT_PDF
        from .ingest import embed_chunks, ingest

        if args.embed_only:
            embed_chunks()
            return 0
        if args.rebuild:
            from .extract import repair_printed_pages
            from .index import build_index, load_meta, load_pages

            pages = repair_printed_pages(load_pages())
            meta = load_meta()
            chunks = build_index(pages, meta)
            print(f"已重建 {len(chunks)} 个片段。")
            if not args.no_embed:
                try:
                    embed_chunks(chunks)
                except Exception as exc:
                    print(f"嵌入跳过（将仅使用关键词检索）：{exc}")
            return 0
        ingest(args.pdf or str(DEFAULT_PDF), with_embeddings=not args.no_embed)
        return 0

    if args.cmd == "ask":
        from .generate import answer_question
        from .index import index_ready

        if not index_ready():
            print("请先运行：python -m pdf_rag ingest", file=sys.stderr)
            return 1
        result = answer_question(args.question, use_embeddings=not args.no_embed)
        print(result.get("answer") or "")
        print()
        for cite in result.get("citations") or []:
            print(f"- {cite.get('location_label')}")
            if cite.get("quote"):
                print(f"  原文：{cite['quote']}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("pdf_rag.app:app", host=args.host, port=args.port, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
