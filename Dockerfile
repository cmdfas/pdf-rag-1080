FROM python:3.9-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pdf_rag ./pdf_rag
COPY static ./static

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "pdf_rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
