from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from .config import (
    FOOTER_RATIO,
    HEADER_RATIO,
    OCR_LANG,
    OCR_MIN_WORD_CONF,
    OCR_PSM,
    OCR_SCALE,
    OCR_WORKERS,
    TESSERACT_CMD,
)

PAGE_NUM_RE = re.compile(r"^(?:[-–—]?\s*)(\d{1,4})(?:\s*[-–—])?$")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章")
SECTION_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百零〇0-9]+[章节]|[0-9]+(?:\.[0-9]+)*\s+\S|[一二三四五六七八九十]+[、．.])"
)
NOISE_RE = re.compile(r"^[\s\\|\/\-_=.,;:'\"`~*<>()\[\]{}]+$")


def _tesseract_tsv(image_path: str, out_base: str) -> str:
    cmd = [
        TESSERACT_CMD,
        image_path,
        out_base,
        "-l",
        OCR_LANG,
        "--psm",
        str(OCR_PSM),
        "-c",
        "preserve_interword_spaces=1",
        "tsv",
    ]
    env = os.environ.copy()
    if "TESSDATA_PREFIX" not in env and os.path.isdir("/usr/local/share/tessdata"):
        env["TESSDATA_PREFIX"] = "/usr/local/share/tessdata"
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    tsv_path = out_base + ".tsv"
    if proc.returncode != 0 or not os.path.exists(tsv_path):
        raise RuntimeError(
            f"tesseract 失败（{proc.returncode}）：{proc.stderr[-400:] if proc.stderr else 'no stderr'}"
        )
    return tsv_path


def _parse_tsv(tsv_path: str, scale: float, page_width: float, page_height: float, pdf_page: int) -> list[dict[str, Any]]:
    with open(tsv_path, encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf") or -1)
        except ValueError:
            conf = -1.0
        y_ratio_px = float(row.get("top") or 0) / (page_height * scale) if page_height and scale else 0.0
        in_margin = y_ratio_px <= HEADER_RATIO or y_ratio_px >= 1.0 - FOOTER_RATIO
        if conf < OCR_MIN_WORD_CONF and not (in_margin and text.isdigit()):
            continue
        key = (row["block_num"], row["par_num"], row["line_num"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)

    lines: list[dict[str, Any]] = []
    for key in order:
        words = grouped[key]
        text = "".join(w.get("text") or "" for w in words).strip()
        text = re.sub(r"[ \t]+", " ", text)
        if not text or NOISE_RE.match(text):
            continue
        lefts, tops, rights, bottoms, confs = [], [], [], [], []
        for w in words:
            left = float(w["left"])
            top = float(w["top"])
            width = float(w["width"])
            height = float(w["height"])
            lefts.append(left)
            tops.append(top)
            rights.append(left + width)
            bottoms.append(top + height)
            try:
                confs.append(float(w.get("conf") or 0))
            except ValueError:
                pass
        x0 = min(lefts) / scale
        top = min(tops) / scale
        x1 = max(rights) / scale
        bottom = max(bottoms) / scale
        y_ratio = top / page_height if page_height else 0.0
        region = "body"
        if y_ratio <= HEADER_RATIO:
            region = "header"
        elif y_ratio >= 1.0 - FOOTER_RATIO:
            region = "footer"
        lines.append(
            {
                "pdf_page": pdf_page,
                "line_no": len(lines) + 1,
                "text": text,
                "bbox": {
                    "x0": round(x0, 2),
                    "top": round(top, 2),
                    "x1": round(x1, 2),
                    "bottom": round(bottom, 2),
                },
                "page_width": round(page_width, 2),
                "page_height": round(page_height, 2),
                "font_size": round((bottom - top), 2),
                "region": region,
                "y_ratio": round(y_ratio, 4),
                "x_ratio": round(x0 / page_width, 4) if page_width else 0.0,
                "conf": round(sum(confs) / len(confs), 1) if confs else 0.0,
            }
        )
    return lines


def _printed_page(lines: list[dict[str, Any]]) -> int | None:
    candidates: list[tuple[int, int]] = []
    for line in lines:
        if line["region"] not in {"header", "footer"}:
            continue
        compact = line["text"].replace(" ", "").replace("O", "0")
        match = PAGE_NUM_RE.match(compact)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 999:
                candidates.append((0 if line["region"] == "header" else 1, value))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _heading(line: dict[str, Any], median_size: float) -> str | None:
    text = line["text"]
    if not text or line["region"] != "body":
        return None
    cleaned = text.replace(" ", "")
    if CHAPTER_RE.search(cleaned) or SECTION_RE.search(cleaned):
        return text[:80]
    if median_size and line["font_size"] >= median_size * 1.35 and 4 <= len(cleaned) <= 40:
        return text
    return None


def _ocr_one_page(payload: tuple[str, int, float]) -> dict[str, Any]:
    pdf_path, pdf_page, scale = payload
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        page = doc[pdf_page - 1]
        width, height = page.get_size()
        image = page.render(scale=scale).to_pil()
    finally:
        doc.close()

    with tempfile.TemporaryDirectory(prefix="pdfocr_") as tmp:
        img_path = os.path.join(tmp, f"p{pdf_page}.png")
        out_base = os.path.join(tmp, f"o{pdf_page}")
        image.save(img_path)
        tsv_path = _tesseract_tsv(img_path, out_base)
        lines = _parse_tsv(tsv_path, scale, float(width), float(height), pdf_page)

    sizes = [ln["font_size"] for ln in lines if ln["region"] == "body" and ln["font_size"]]
    median_size = 0.0
    if sizes:
        ordered = sorted(sizes)
        median_size = ordered[len(ordered) // 2]
    headings = []
    for ln in lines:
        title = _heading(ln, median_size)
        if title:
            headings.append(
                {"line_no": ln["line_no"], "text": title, "font_size": ln["font_size"]}
            )
    return {
        "pdf_page": pdf_page,
        "width": round(float(width), 2),
        "height": round(float(height), 2),
        "printed_page": _printed_page(lines),
        "line_count": len(lines),
        "headings": headings,
        "lines": lines,
        "scale": scale,
    }


def extract_outline(pdf_path: str) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return []
    reader = PdfReader(pdf_path)
    outlines: list[dict[str, Any]] = []

    def walk(items, level: int = 0) -> None:
        if not items:
            return
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            title = str(getattr(item, "title", None) or item)
            page_no = None
            try:
                page_no = reader.get_destination_page_number(item) + 1
            except Exception:
                page_no = None
            outlines.append({"title": title, "pdf_page": page_no, "level": level})

    try:
        walk(reader.outline)
    except Exception:
        return outlines
    return outlines


def extract_pdf(pdf_path: str, progress=None, workers: int | None = None) -> list[dict[str, Any]]:
    if not shutil.which(TESSERACT_CMD):
        raise RuntimeError("未找到 tesseract，请先安装：brew install tesseract tesseract-lang")

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    try:
        total = len(doc)
    finally:
        doc.close()

    jobs = [(pdf_path, i, OCR_SCALE) for i in range(1, total + 1)]
    n_workers = max(1, workers or OCR_WORKERS)
    pages: list[dict[str, Any] | None] = [None] * total
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_ocr_one_page, job): job[1] for job in jobs}
        for fut in as_completed(futures):
            pdf_page = futures[fut]
            record = fut.result()
            pages[pdf_page - 1] = record
            done += 1
            if progress:
                progress(done, total)

    out: list[dict[str, Any]] = []
    for record in pages:
        assert record is not None
        out.append(record)
    return repair_printed_pages(out)


def repair_printed_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fix OCR page-number mistakes using the stable PDF-to-book offset."""
    offsets: list[int] = []
    for rec in pages:
        printed = rec.get("printed_page")
        if not printed:
            continue
        offset = rec["pdf_page"] - int(printed)
        if 3 <= offset <= 10:
            offsets.append(offset)
    median_offset = 5
    if offsets:
        median_offset = sorted(offsets)[len(offsets) // 2]
    for rec in pages:
        expected = rec["pdf_page"] - median_offset
        printed = rec.get("printed_page")
        if expected < 1:
            rec["printed_page"] = None
            continue
        if printed is None or abs(int(printed) - expected) > 2:
            rec["printed_page"] = expected
    return pages
