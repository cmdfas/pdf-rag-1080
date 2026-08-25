from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import BOOK_LINK, DEFAULT_PDF, STATIC_DIR
from .generate import answer_question
from .config import EMBED_PATH
from .index import index_ready, load_meta

app = FastAPI(title="教材 RAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_embeddings: bool = True


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    pdf = BOOK_LINK if BOOK_LINK.exists() else DEFAULT_PDF
    meta = load_meta() if index_ready() else None
    return {
        "ok": True,
        "index_ready": index_ready(),
        "pdf_exists": Path(pdf).exists(),
        "page_count": (meta or {}).get("page_count"),
        "title": (meta or {}).get("title"),
        "embeddings_ready": EMBED_PATH.exists(),
    }


@app.get("/api/meta")
def meta() -> dict:
    if not index_ready():
        raise HTTPException(409, "索引尚未建立，请先运行：python -m pdf_rag.ingest")
    return load_meta()


@app.get("/book.pdf")
def book_pdf() -> FileResponse:
    path = BOOK_LINK if BOOK_LINK.exists() else DEFAULT_PDF
    if not Path(path).exists():
        raise HTTPException(404, "找不到教材 PDF")
    return FileResponse(path, media_type="application/pdf", filename="book.pdf")


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    if not index_ready():
        raise HTTPException(409, "索引尚未建立，请先运行：python -m pdf_rag.ingest")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "问题不能为空")
    return answer_question(question, use_embeddings=req.use_embeddings)
