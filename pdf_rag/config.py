from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
STATIC_DIR = ROOT / "static"

DEFAULT_PDF = Path(
    "/Users/sunny/Downloads/软件工程 - 01 VIP资料/13005软件工程_2024年版(张琼声).pdf"
)
BOOK_LINK = DATA_DIR / "book.pdf"

PAGES_PATH = INDEX_DIR / "pages.jsonl"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
META_PATH = INDEX_DIR / "meta.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"
EMBED_PATH = INDEX_DIR / "embeddings.npy"

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE_URL = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-4.6")
XAI_EMBED_MODEL = os.environ.get("XAI_EMBED_MODEL", "grok-embedding-large")

# Grok CLI chat proxy（官方文档：auth.json + cli-chat-proxy）
GROK_CLI_PROXY_URL = os.environ.get(
    "GROK_CLI_PROXY_URL", "https://cli-chat-proxy.grok.com/v1"
)
GROK_CLI_MODEL = os.environ.get("GROK_CLI_MODEL", "grok-build")
GROK_CLI_CLIENT_VERSION = os.environ.get("GROK_CLI_CLIENT_VERSION", "0.1.202")
GROK_AUTH_JSON = Path(
    os.environ.get("GROK_AUTH_JSON", str(Path.home() / ".grok" / "auth.json"))
).expanduser()
CLI_SIGNIN_KEY = "https://accounts.x.ai/sign-in"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_EMBED_URL = os.environ.get(
    "DASHSCOPE_EMBED_URL",
    "https://llm-e4ealg1c1ytymvji.cn-beijing.maas.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
)
DASHSCOPE_EMBED_MODEL = os.environ.get("DASHSCOPE_EMBED_MODEL", "qwen3.7-text-embedding")
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "10"))

CHUNK_SIZE = 700
CHUNK_OVERLAP_LINES = 2
RETRIEVE_K = 20
EMBED_TOP_K = 10
HYBRID_K = 12

# Visual line clustering: words whose top differs by less than this are one line.
LINE_Y_TOLERANCE = 3.5
HEADER_RATIO = 0.07
FOOTER_RATIO = 0.08

OCR_SCALE = 2.8
OCR_LANG = "chi_sim+eng"
OCR_PSM = 4
OCR_MIN_WORD_CONF = 40
OCR_WORKERS = 4
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "tesseract")
