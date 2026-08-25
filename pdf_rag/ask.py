from __future__ import annotations

import json
import sys

from .generate import answer_question
from .index import index_ready


def main() -> int:
    if not index_ready():
        print("请先运行：python -m pdf_rag ingest", file=sys.stderr)
        return 1
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("用法：python -m pdf_rag.ask 什么是软件工程？", file=sys.stderr)
        return 1
    result = answer_question(question)
    print(result.get("answer") or "")
    print()
    for cite in result.get("citations") or []:
        print(f"- {cite.get('location_label')}")
        if cite.get("quote"):
            print(f"  原文：{cite['quote']}")
    if "--json" in sys.argv:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
