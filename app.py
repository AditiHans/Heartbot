import os, re, asyncio, logging
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Keep memory small
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CHROMADB_DISABLE_ONNX_RUNTIME", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import httpx
import chromadb

# ---------- Required Cloud Config ----------
CHROMA_HOST     = os.getenv("CHROMA_HOST")      # e.g. "api.trychroma.com"
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_TOKEN    = os.getenv("CHROMA_API_TOKEN")   # header: x-chroma-token

HF_API_TOKEN    = os.getenv("HF_API_TOKEN")
HF_QA_MODEL     = os.getenv("HF_QA_MODEL", "deepset/roberta-base-squad2")

# Retrieval config
COLLECTION      = os.environ.get("COLLECTION", "heart_faq")

# Answering thresholds
TOP_K_DEFAULT   = int(os.environ.get("TOP_K", "4"))
FETCH_K_DEF     = int(os.environ.get("FETCH_K", "12"))   # pull a few more, helps recall
MIN_TOKENS      = int(os.environ.get("MIN_TOKENS", "1")) # allow short spans like "Dyspnea"
MAX_CTX_CHARS   = int(os.environ.get("MAX_CTX_CHARS", "2000"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT_SECS", "25"))

# Optional debug: include top ctx previews + raw HF outputs in response
DEBUG_CTX = os.getenv("DEBUG_CTX", "0") == "1"

# ---------- FastAPI ----------
app = FastAPI(title="Heartify RAG API (Cloud-only, lite)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log = logging.getLogger("uvicorn.error")

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# ---------- Lazy singletons ----------
_collection = None
_http = None

def _missing_env() -> List[str]:
    missing = []
    if not CHROMA_HOST:     missing.append("CHROMA_HOST")
    if not CHROMA_TENANT:   missing.append("CHROMA_TENANT")
    if not CHROMA_DATABASE: missing.append("CHROMA_DATABASE")
    if not CHROMA_TOKEN:    missing.append("CHROMA_API_TOKEN")
    if not HF_API_TOKEN:    missing.append("HF_API_TOKEN")
    return missing

async def _retry(fn, tries=3, delay=1.0):
    last = None
    for _ in range(tries):
        try:
            return await fn()
        except Exception as e:
            last = e
            await asyncio.sleep(delay)
    raise last

async def _get_http():
    global _http
    if _http:
        return _http
    # small timeouts to avoid memory leaks/hangs
    _http = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=10.0))
    return _http

async def _get_collection():
    missing = _missing_env()
    if missing:
        raise HTTPException(status_code=503, detail={"error": "Cloud not configured", "missing_env": missing})

    global _collection
    if _collection is not None:
        return _collection

    def _open_cloud():
        client = chromadb.HttpClient(
            host=CHROMA_HOST,
            ssl=True,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            headers={"x-chroma-token": CHROMA_TOKEN},
        )
        return client.get_or_create_collection(COLLECTION)

    async def _coro(): return _open_cloud()
    _collection = await _retry(_coro)
    return _collection

# ---------- Hugging Face Inference (QA) ----------
async def _hf_answer(question: str, context: str) -> dict:
    try:
        http = await _get_http()
        url = f"https://api-inference.huggingface.co/models/{HF_QA_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": {"question": question, "context": context}}

        log.info(f"Sending request to Hugging Face: {payload}")

        for _ in range(3):  # retry a couple times (models may “warm up”)
            r = await http.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                data = r.json()
                log.info(f"Hugging Face response: {data}")
                if isinstance(data, list) and data:
                    data = data[0]
                if isinstance(data, dict) and "answer" in data:
                    return {"answer": (data.get("answer") or "").strip(), "score": float(data.get("score") or 0.0)}
                if isinstance(data, str):
                    return {"answer": data.strip(), "score": 0.0}
            if r.status_code in (503, 524, 408):
                await asyncio.sleep(1.5)
                continue
            try:
                log.warning("HF error %s: %s", r.status_code, r.json())
            except Exception:
                log.warning("HF error %s: %s", r.status_code, r.text)
            return {"answer": "", "score": 0.0}
        return {"answer": "", "score": 0.0}
    except Exception as e:
        log.exception("Error while contacting Hugging Face")
        return {"answer": "Unexpected error.", "score": 0.0}
