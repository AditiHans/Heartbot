# app.py — Heartify RAG API (cloud-only, robust w/ Gemini fallback)
import os, re, asyncio, logging
from typing import List, Optional

# NEW: load .env early
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)  # <— loads .env if present

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---- Stability flags (set BEFORE heavy imports) ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CHROMADB_DISABLE_ONNX_RUNTIME", "1")

import httpx
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# ---------- Required Cloud Config ----------
CHROMA_HOST     = os.getenv("CHROMA_HOST")      # e.g. "api.trychroma.com"
CHROMA_TENANT   = os.getenv("CHROMA_TENANT")    # provided by trychroma
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")  # provided by trychroma
CHROMA_TOKEN    = os.getenv("CHROMA_API_TOKEN") # your x-chroma-token

# Gemini fallback
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

COLLECTION      = os.environ.get("COLLECTION", "heart_faq")
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_PATH      = os.environ.get("MODEL_PATH",  "Adi9818/taskB_continual")  # or your HF repo

TOP_K_DEFAULT   = int(os.environ.get("TOP_K", "4"))
FETCH_K_DEF     = int(os.environ.get("FETCH_K", "10"))
MIN_TOKENS      = int(os.environ.get("MIN_TOKENS", "6"))
MIN_CONF        = float(os.environ.get("MIN_CONF", "0.25"))
MAX_CTX_CHARS   = int(os.environ.get("MAX_CTX_CHARS", "1400"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT_SECS", "25"))

# CPU-first (cloud free tiers rarely have GPU)
_DEVICE = -1
if torch.cuda.is_available():
    _DEVICE = 0

# ---------- FastAPI ----------
app = FastAPI(title="Heartify RAG API (Cloud)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
log = logging.getLogger("uvicorn.error")

# ---------- Global guards ----------
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

# ---------- Lazy singletons + retry ----------
_collection = None
_embedder = None
_qa = None
_http = None  # for Gemini calls

def _missing_cloud_vars() -> List[str]:
    missing = []
    if not CHROMA_HOST:     missing.append("CHROMA_HOST")
    if not CHROMA_TENANT:   missing.append("CHROMA_TENANT")
    if not CHROMA_DATABASE: missing.append("CHROMA_DATABASE")
    if not CHROMA_TOKEN:    missing.append("CHROMA_API_TOKEN")  # matches the name we read
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
    timeout = httpx.Timeout(15.0, connect=10.0)
    _http = httpx.AsyncClient(timeout=timeout)
    return _http

async def _get_collection():
    missing = _missing_cloud_vars()
    if missing:
        raise HTTPException(
            status_code=503,
            detail={"error": "Chroma Cloud not configured", "missing_env": missing},
        )

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
        # NOTE: Cloud collection must already be populated (build upstream).
        return client.get_or_create_collection(COLLECTION)

    async def _coro():
        return _open_cloud()

    _collection = await _retry(_coro)
    return _collection

async def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    async def _coro():
        return SentenceTransformer(EMBED_MODEL)

    _embedder = await _retry(_coro)
    return _embedder

async def _get_qa():
    global _qa
    if _qa is not None:
        return _qa

    async def _coro():
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        mdl = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
        return pipeline(
            "question-answering",
            model=mdl,
            tokenizer=tok,
            device=_DEVICE,  # -1 CPU; 0 GPU if available
            handle_impossible_answer=True,
            max_answer_len=64,
            align_to_words=True,
        )
    _qa = await _retry(_coro)
    return _qa

# ---------- Gemini fallback ----------
async def _gemini_answer(question: str, context: str) -> str:
    """Call Gemini with strict 'answer from context only' prompt."""
    if not GEMINI_API_KEY:
        return ""  # no key configured; skip fallback

    http = await _get_http()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    system_rules = (
        "You are a concise, factual medical assistant.\n"
        "Answer ONLY using the provided context below.\n"
        "- Use brief sentences or bullet points.\n"
        "- No greetings or sign-offs.\n"
        "- If the context lacks the answer, reply exactly: "
        "'Not enough information in the provided context.'"
    )
    payload = {
        "contents": [{
            "parts": [{
                "text": f"{system_rules}\n\nQuestion: {question}\n\nContext:\n{context}"
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 220,
        }
    }

    try:
        r = await http.post(url, json=payload, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            # soft fail — do not crash API
            try:
                log.warning("Gemini fallback HTTP %s: %s", r.status_code, r.json())
            except Exception:
                log.warning("Gemini fallback HTTP %s: %s", r.status_code, r.text)
            return ""
        data = r.json()
        # Extract first candidate text
        text = ""
        for cand in (data.get("candidates") or []):
            parts = (((cand.get("content") or {}).get("parts")) or [])
            for p in parts:
                if "text" in p:
                    text += p["text"]
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        log.warning("Gemini fallback error: %s", e)
        return ""

# ---------- Helpers ----------
NOISE_RE = re.compile(
    r"\b(dear|hello|hi)\b|thank you|thanks|best regards|regards|yours|dr\.|\bmd\b|"
    r"consult|please rate|assist you further|ask.*(doctor|hcm)|wish you",
    re.I,
)

def clean_context(text: str) -> str:
    t = re.sub(r"(?i)regards.*", "", text)
    t = re.sub(r"(?i)best regards.*", "", t)
    t = re.sub(r"(?i)thank.*", "", t)
    t = re.sub(r"(?i)dr\.\s+[A-Z][a-z].*", "", t)
    return re.sub(r"\s+", " ", t).strip()

def score_doc(doc: str, q_words: set, dist: float, source: str) -> float:
    kw = sum(1 for w in q_words if w in doc.lower())
    bonus = 1.0 if "taska_ctx.json" in (source or "").lower() else 0.0
    penalty = 1.5 if NOISE_RE.search(doc) else 0.0
    return kw + bonus - penalty - float(dist)

async def _retrieve(question: str, fetch_k: int, top_k: int):
    collection = await _get_collection()
    embedder = await _get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=max(fetch_k, top_k),
        include=["documents", "metadatas", "distances"],
    )
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    return docs, metas, dists

# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: Optional[int] = Field(default=TOP_K_DEFAULT, ge=1, le=10)
    fetch_k: Optional[int] = Field(default=FETCH_K_DEF, ge=1, le=50)
    extra_contexts: Optional[List[str]] = None  # per-request, not persisted

# ---------- Endpoints ----------
@app.get("/health")
async def health():
    missing = _missing_cloud_vars()
    ok = len(missing) == 0
    return {
        "ok": ok,
        "missing_env": missing,
        "device": ("gpu" if _DEVICE == 0 else "cpu"),
        "collection": COLLECTION,
        "embed_model": EMBED_MODEL,
        "model_path": MODEL_PATH,
        "gemini": bool(GEMINI_API_KEY),
    }

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/ask")
async def ask(req: AskRequest):
    async def _work():
        q = req.question.strip()
        if not q:
            raise HTTPException(status_code=422, detail="Question must be non-empty.")

        # 1) Retrieve from Chroma Cloud
        docs, metas, dists = await _retrieve(q, req.fetch_k or FETCH_K_DEF, req.top_k or TOP_K_DEFAULT)

        # 2) Add ephemeral user contexts (not persisted)
        extra_docs = req.extra_contexts or []
        docs  += extra_docs
        metas += [{"source": "ephemeral"} for _ in extra_docs]
        dists += [0.25 for _ in extra_docs]

        if not docs:
            return {"answer": "Not enough information in the provided context.", "confidence": 0.0, "sources": []}

        # 3) Clean, score, keep top_k
        q_words = {w for w in re.findall(r"[A-Za-z]+", q.lower()) if len(w) > 3}
        scored = []
        for d, m, dist in zip(docs, metas, dists):
            d_clean = clean_context(d or "")
            if len(d_clean) < 60:
                continue
            sc = score_doc(d_clean, q_words, dist, (m or {}).get("source", ""))
            scored.append((sc, d_clean, (m or {}).get("source", "unknown"), float(dist)))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = scored[: max(req.top_k or TOP_K_DEFAULT, 1)]
        if not keep:
            return {"answer": "Not enough information in the provided context.", "confidence": 0.0, "sources": []}

        # 4) Extractive QA over top docs
        qa = await _get_qa()
        best = {"text": "", "score": -1.0, "len": 0}
        merged_context = ""
        for i, (_, d_clean, _, _) in enumerate(keep, 1):
            ctx = d_clean[:MAX_CTX_CHARS]  # truncate for stability
            merged_context += (("\n\n" if merged_context else "") + ctx)
            out = qa({"question": q, "context": ctx})
            ans = re.sub(r"\s+", " ", out.get("answer", "")).strip()
            score = float(out.get("score", 0.0))
            tok_len = len(ans.split())
            if tok_len < MIN_TOKENS:
                continue
            if (score > best["score"]) or (abs(score - best["score"]) < 1e-6 and tok_len > best["len"]):
                best = {"text": ans, "score": score, "len": tok_len}

        # 4b) Gemini fallback if low confidence or empty span
        if (not best["text"]) or (best["score"] < MIN_CONF):
            if GEMINI_API_KEY:
                # keep merged context short to stay under token limits
                if len(merged_context) > MAX_CTX_CHARS:
                    merged_context = merged_context[:MAX_CTX_CHARS]
                g_ans = await _gemini_answer(q, merged_context)
                if g_ans:
                    sources = [{"source": s, "distance": round(d, 4)} for _, _, s, d in keep]
                    # keep response shape the same (no extra keys)
                    return {
                        "answer": g_ans,
                        "confidence": round(max(best["score"], 0.0), 4),  # keep extractive score for continuity
                        "sources": sources,
                    }

        sources = [{"source": s, "distance": round(d, 4)} for _, _, s, d in keep]
        return {
            "answer": best["text"] or "Not enough information in the provided context.",
            "confidence": round(max(best["score"], 0.0), 4),
            "sources": sources,
        }

    try:
        return await asyncio.wait_for(_work(), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out, try narrowing your question.")
