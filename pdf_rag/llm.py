from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from .config import (
    CLI_SIGNIN_KEY,
    DASHSCOPE_API_KEY,
    DASHSCOPE_EMBED_MODEL,
    DASHSCOPE_EMBED_URL,
    EMBED_BATCH_SIZE,
    GROK_AUTH_JSON,
    GROK_CLI_CLIENT_VERSION,
    GROK_CLI_MODEL,
    GROK_CLI_PROXY_URL,
    XAI_API_KEY,
    XAI_BASE_URL,
    XAI_EMBED_MODEL,
    XAI_MODEL,
)


def _cli_session_token() -> str:
    """Read the Grok CLI session token from ~/.grok/auth.json.

    Official path is ``https://accounts.x.ai/sign-in``. Some CLI builds store
    the same token under ``https://auth.x.ai::<uuid>``.
    """
    path = GROK_AUTH_JSON
    if not path.exists():
        raise RuntimeError(f"找不到 {path}，请先运行：grok login")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"无法读取 {path}") from exc

    preferred = payload.get(CLI_SIGNIN_KEY)
    if isinstance(preferred, dict) and preferred.get("key"):
        return str(preferred["key"])

    for entry in payload.values():
        if isinstance(entry, dict) and entry.get("key"):
            return str(entry["key"])
    raise RuntimeError(f"{path} 里没有 session token，请运行：grok login")


def _api_key() -> str:
    return os.environ.get("XAI_API_KEY") or XAI_API_KEY


def _client() -> OpenAI:
    key = _api_key()
    if not key:
        raise RuntimeError("未设置 XAI_API_KEY。")
    return OpenAI(api_key=key, base_url=XAI_BASE_URL, timeout=120.0)


def _dashscope_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY


def _parse_dashscope_embeddings(payload: dict[str, Any], n_input: int) -> list[list[float]]:
    output = payload.get("output") or payload
    items = output.get("embeddings") if isinstance(output, dict) else None
    if not items and isinstance(payload.get("data"), list):
        items = payload["data"]
    if not items:
        raise RuntimeError(f"嵌入响应缺少 embeddings：{list(payload)[:8]}")
    ordered = sorted(
        items,
        key=lambda it: int(it.get("text_index", it.get("index", 0))),
    )
    vectors = [list(it["embedding"]) for it in ordered]
    if len(vectors) != n_input:
        raise RuntimeError(f"嵌入条数不符：期望 {n_input}，实际 {len(vectors)}")
    return vectors


def _embed_dashscope(texts: list[str], on_batch=None) -> list[list[float]]:
    key = _dashscope_key()
    if not key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY")
    vectors: list[list[float]] = []
    batch_size = max(1, EMBED_BATCH_SIZE)
    timeout = httpx.Timeout(60.0)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout) as client:
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            payload = _post_embed(client, headers, batch)
            vectors.extend(_parse_dashscope_embeddings(payload, len(batch)))
            done = min(i + batch_size, len(texts))
            if done == len(texts) or done % (batch_size * 5) == 0 or i == 0:
                print(f"  向量 {done}/{len(texts)}", flush=True)
            if on_batch:
                on_batch(vectors)
            time.sleep(0.4)
    return vectors


def _post_embed(client: httpx.Client, headers: dict[str, str], batch: list[str]) -> dict[str, Any]:
    delay = 2.0
    last_error = "unknown"
    for attempt in range(8):
        resp = client.post(
            DASHSCOPE_EMBED_URL,
            headers=headers,
            json={
                "model": DASHSCOPE_EMBED_MODEL,
                "input": {"texts": batch},
            },
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text[:240]}
        if resp.status_code < 400:
            return payload
        message = (
            payload.get("message")
            or payload.get("code")
            or str(payload)[:240]
        )
        last_error = f"HTTP {resp.status_code}：{message}"
        if resp.status_code in {429, 500, 502, 503, 504}:
            print(f"  限流/重试 {attempt + 1}/8，等待 {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 1.8, 45)
            continue
        raise RuntimeError(f"嵌入接口 {last_error}")
    raise RuntimeError(f"嵌入接口多次失败：{last_error}")


EMBED_CANDIDATES = (
    XAI_EMBED_MODEL,
    "grok-embedding-large",
    "grok-embedding-small",
)


def embed_texts(texts: list[str], model: str | None = None, on_batch=None) -> list[list[float]]:
    if not texts:
        return []
    if not model and _dashscope_key():
        return _embed_dashscope(texts, on_batch=on_batch)
    client = _client()
    models = [model] if model else list(EMBED_CANDIDATES)
    last_error: Exception | None = None
    for model_name in models:
        try:
            vectors: list[list[float]] = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(model=model_name, input=batch)
                ordered = sorted(resp.data, key=lambda d: d.index)
                vectors.extend([list(d.embedding) for d in ordered])
            return vectors
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"嵌入接口不可用：{last_error}")


def _delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return str(event.get("content") or "")
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    if isinstance(delta, dict) and delta.get("content"):
        return str(delta["content"])
    message = choice.get("message") or {}
    if isinstance(message, dict) and message.get("content"):
        return str(message["content"])
    return str(choice.get("text") or "")


def _cli_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
    """Call the Grok CLI chat proxy exactly as documented: session token + stream."""
    token = _cli_session_token()
    model_name = model or GROK_CLI_MODEL
    url = GROK_CLI_PROXY_URL.rstrip("/") + "/chat/completions"
    body = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": 0.1,
    }
    last_error = "unknown"
    for attempt in range(4):
        try:
            return _cli_chat_once(url, token, model_name, body)
        except Exception as exc:
            last_error = str(exc)
            if attempt == 3:
                break
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(last_error)


def _cli_chat_once(url: str, token: str, model_name: str, body: dict[str, Any]) -> str:
    pieces: list[str] = []
    # Official docs use curl; httpx CONNECT to this host is unreliable behind mixed HTTP/SOCKS proxies.
    cmd = [
        "curl",
        "-sS",
        "-N",
        "--max-time",
        "180",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "X-XAI-Token-Auth: xai-grok-cli",
        "-H",
        f"x-grok-model-override: {model_name}",
        "-H",
        f"x-grok-client-version: {GROK_CLI_CLIENT_VERSION}",
        "-H",
        "x-grok-client-identifier: grok-shell",
        "-H",
        f"User-Agent: grok-shell/{GROK_CLI_CLIENT_VERSION}",
        "-d",
        json.dumps(body, ensure_ascii=False),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    raw_lines: list[str] = []
    for line in proc.stdout:
        raw_lines.append(line)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("data:"):
            data = stripped[5:].strip()
        elif stripped.startswith("{"):
            data = stripped
        else:
            continue
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event.get("error"):
            proc.kill()
            raise RuntimeError(f"CLI 代理错误：{event['error']}")
        piece = _delta_text(event)
        if piece:
            pieces.append(piece)
    stderr = proc.communicate()[1]
    if proc.returncode not in (0, None) and not pieces:
        err = (stderr or "").strip()[:400]
        raise RuntimeError(f"CLI 代理 curl 失败：{err or proc.returncode}")
    text = "".join(pieces).strip()
    if not text:
        preview = "".join(raw_lines)[:400]
        if "outdated" in preview or "Grok CLI version" in preview:
            raise RuntimeError(preview)
        hint = " Token 可能过期，请运行：grok login" if "401" in preview or "403" in preview else ""
        raise RuntimeError(f"CLI 代理返回空内容。{hint} {preview}")
    return text


def _parse_json_content(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def chat_backend() -> str:
    """Production should set XAI_API_KEY (official API). Local CLI login is the fallback."""
    if _api_key():
        return "api"
    return "cli-proxy"


def _api_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
    client = _client()
    resp = client.chat.completions.create(
        model=model or XAI_MODEL,
        temperature=0.1,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def chat_json(system: str, user: str, model: str | None = None) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if _api_key():
        content = _api_chat(messages, model=model)
    else:
        content = _cli_chat(messages, model=model)
    return _parse_json_content(content)
