from __future__ import annotations

from typing import Any

from .index import load_embeddings
from .llm import chat_json, embed_texts
from .retrieve import hybrid_search

SYSTEM = """你是《软件工程》（2024年版，张琼声）教材的答题助手。
只根据提供的教材片段作答，不要使用教材以外的知识去编造定义或考点。
教材文本由扫描版 OCR 得到，可能有个别错字（如“浊布”应为“瀑布”），也可能把一个词拆在相邻两行。答案里可以按上下文纠正明显错字并拼回完整词语，但 citations.quote 必须用片段里的原文。
优先引用带“……是……”的定义句、条目和原理段落；章节导读、学习目标、课后习题只作补充。
片段已按相关度排序，【最相关】必须优先采用。只要其中出现“XX是……一门学科/过程/模型/集合”这类句子，就必须当作定义写进 answer，禁止回答“未找到”。
只有当所有片段都没有对应知识点时，才说教材片段中未找到，并指出最接近的内容。

必须输出 JSON，字段：
{
  "answer": "完整中文答案，可分点，但不要空话",
  "not_found": false,
  "citations": [
    {
      "chunk_id": "C0001",
      "quote": "从片段中原样摘录的短句，用于定位",
      "pdf_page": 12,
      "printed_page": 1,
      "line_start": 8,
      "line_end": 14
    }
  ]
}

规则：
1. citations 只能引用提供片段中真实存在的 chunk_id 和原文。
2. 每个关键论断都要有 citation。
3. printed_page 是教材页码，pdf_page 是 PDF 页码。
4. line_start/line_end 使用片段元数据中的行号，不要自己估。
5. quote 必须是片段里出现过的连续原文，尽量短（不超过 80 字）。
"""


def _format_hit(hit: dict[str, Any], rank: int = 0) -> str:
    printed = hit.get("printed_page")
    printed_end = hit.get("printed_page_end")
    page_label = f"教材第 {printed} 页" if printed else "教材页码未知"
    if printed and printed_end and printed_end != printed:
        page_label = f"教材第 {printed}–{printed_end} 页"
    heading = hit.get("heading") or hit.get("chapter") or ""
    mark = "【最相关】" if rank == 1 else f"【相关{rank}】"
    loc = (
        f"{mark} [{hit['id']}] {page_label}｜PDF 第 {hit['pdf_page']} 页"
        f"｜第 {hit['line_start']}–{hit['line_end']} 行"
    )
    if heading:
        loc += f"｜章节：{heading}"
    return f"{loc}\n{hit['text']}"


def _enrich_citations(raw: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {h["id"]: h for h in hits}
    citations = []
    for cite in raw.get("citations") or []:
        chunk = by_id.get(cite.get("chunk_id"))
        if not chunk:
            continue
        quote = (cite.get("quote") or "").strip()
        pdf_page = cite.get("pdf_page") or chunk["pdf_page"]
        line_start = cite.get("line_start") or chunk["line_start"]
        line_end = cite.get("line_end") or chunk["line_end"]
        highlight = []
        for box in chunk.get("bboxes") or []:
            if box["pdf_page"] != pdf_page:
                continue
            if line_start <= box["line_no"] <= line_end:
                highlight.append(box)
        if quote:
            # Prefer boxes whose surrounding chunk text contains the quote;
            # shrink to lines that actually include quote characters.
            narrowed = []
            chunk_text_lines = chunk.get("raw_lines") or chunk["text"].split("\n")
            # Map sequential body lines in chunk to bboxes (same order).
            for text_line, box in zip(chunk_text_lines, chunk.get("bboxes") or []):
                if quote in text_line or (len(quote) >= 4 and quote[:8] in text_line):
                    narrowed.append(box)
            if not narrowed and quote:
                compact_quote = quote.replace(" ", "")
                joined = ""
                run: list[dict[str, Any]] = []
                for text_line, box in zip(chunk_text_lines, chunk.get("bboxes") or []):
                    joined += text_line.replace(" ", "")
                    run.append(box)
                    if compact_quote in joined:
                        narrowed = run
                        break
                    if len(joined) > len(compact_quote) + 40:
                        joined = text_line.replace(" ", "")
                        run = [box]
            if narrowed:
                highlight = narrowed
                pdf_page = highlight[0]["pdf_page"]
                line_start = highlight[0]["line_no"]
                line_end = highlight[-1]["line_no"]

        printed_page = chunk.get("printed_page")
        for span in chunk.get("pages") or []:
            if span["pdf_page"] == pdf_page:
                printed_page = span.get("printed_page")
                break

        first = highlight[0] if highlight else None
        citations.append(
            {
                "chunk_id": chunk["id"],
                "quote": quote,
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "line_start": line_start,
                "line_end": line_end,
                "heading": chunk.get("heading") or chunk.get("chapter"),
                "score": chunk.get("score"),
                "bbox": {
                    "x0": first["x0"],
                    "top": first["top"],
                    "x1": first["x1"],
                    "bottom": first["bottom"],
                    "page_width": first["page_width"],
                    "page_height": first["page_height"],
                }
                if first
                else None,
                "highlights": highlight,
                "location_label": _location_label(
                    printed_page, pdf_page, line_start, line_end, first
                ),
            }
        )
    raw["citations"] = citations
    raw.setdefault("answer", "")
    raw.setdefault("not_found", False)
    return raw


def _location_label(
    printed_page: int | None,
    pdf_page: int,
    line_start: int,
    line_end: int,
    box: dict[str, Any] | None,
) -> str:
    parts = []
    if printed_page:
        parts.append(f"教材第 {printed_page} 页")
    parts.append(f"PDF 第 {pdf_page} 页")
    if line_start == line_end:
        parts.append(f"第 {line_start} 行")
    else:
        parts.append(f"第 {line_start}–{line_end} 行")
    if box and box.get("page_height") and box.get("page_width"):
        top_pct = round(100.0 * box["top"] / box["page_height"], 1)
        left_pct = round(100.0 * box["x0"] / box["page_width"], 1)
        parts.append(f"约距页顶 {top_pct}%、距页左 {left_pct}%")
    return "｜".join(parts)


def answer_question(question: str, use_embeddings: bool = True) -> dict[str, Any]:
    query_vec = None
    if use_embeddings and load_embeddings() is not None:
        try:
            query_vec = embed_texts([question])[0]
            import numpy as np

            query_vec = np.array(query_vec, dtype=np.float64)
        except Exception:
            query_vec = None
    hits = hybrid_search(question, query_vec=query_vec)
    if not hits:
        return {
            "answer": "索引中没有检索到相关段落，请先确认教材已完成 ingest。",
            "not_found": True,
            "citations": [],
            "retrieved": [],
        }
    context = "\n\n".join(_format_hit(h, i) for i, h in enumerate(hits, start=1))
    user = f"问题：{question}\n\n请根据下列教材片段作答，尤其是【最相关】。\n\n教材片段：\n{context}"
    raw = chat_json(SYSTEM, user)
    result = _enrich_citations(raw, hits)
    result["retrieved"] = [
        {
            "id": h["id"],
            "pdf_page": h["pdf_page"],
            "printed_page": h.get("printed_page"),
            "line_start": h["line_start"],
            "line_end": h["line_end"],
            "heading": h.get("heading"),
            "score": h.get("score"),
            "preview": h["text"][:160].replace("\n", " "),
        }
        for h in hits
    ]
    return result
