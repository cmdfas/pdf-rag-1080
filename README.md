# 软件工程教材 RAG

把《软件工程》（2024 年版，张琼声）做成可提问的检索问答：答案只来自教材原文，并标出 **教材页码、PDF 页码、行号、页内位置**。点引用可跳到 PDF 对应页并高亮原文。

## 准备

需要本机已安装 Tesseract（含 `chi_sim`），本书是扫描版，必须 OCR。

```bash
brew install tesseract tesseract-lang
cd /Users/sunny/Downloads/pdf-rag-1080
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
grok login    # 答题走 CLI 官方代理，读 ~/.grok/auth.json
```

默认 PDF：

`/Users/sunny/Downloads/软件工程 - 01 VIP资料/13005软件工程_2024年版(张琼声).pdf`

## 建立索引

```bash
python -m pdf_rag ingest
```

只做关键词检索、不调用嵌入模型：

```bash
python -m pdf_rag ingest --no-embed
```

已有 OCR 索引、只补向量（阿里云 `qwen3.7-text-embedding`）：

```bash
# .env 里设置 DASHSCOPE_API_KEY
python -m pdf_rag ingest --embed-only
```

## 提问

网页（推荐，可对照原文）：

```bash
python -m pdf_rag serve
```

打开 http://127.0.0.1:8000

部署到服务器或 Docker 见 [docs/部署.md](docs/部署.md)。生产请配置 `XAI_API_KEY`，不要用本机 `grok login`。

命令行：

```bash
python -m pdf_rag ask "什么是软件生存周期？"
```

答题默认走 Grok CLI 代理。官方 API（`api.x.ai`）与 CLI 代理（`cli-chat-proxy.grok.com`）的差别、curl 示例和必带头见 [docs/模型调用.md](docs/模型调用.md)。登录态约 7 天过期，失效后重新 `grok login`。

## 答案里会给出什么

- 基于教材原文的回答
- 每条依据包含：教材页码、PDF 页码、第几行到第几行、距页顶/页左的大致位置
- 网页里点击依据，右侧 PDF 会跳转并高亮对应行
