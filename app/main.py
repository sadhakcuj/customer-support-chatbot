"""Customer support chatbot API: grounded retrieval, optional LLM, sentiment triage, human handoff."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import escalation
from app.config import Settings, get_settings
from app.llm import LLMAnswerer
from app.retriever import Retriever
from app.schemas import ChatRequest, ChatResponse, FeedbackRequest, Source
from app.sessions import SessionStore
from app.triage import assess

APP_DIR = Path(__file__).resolve().parent
settings = get_settings()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for key in ("request_id", "path", "status", "ms", "mode", "escalated", "reason", "score"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.getLogger().handlers = [_handler]
logging.getLogger().setLevel(settings.log_level.upper())
log = logging.getLogger("bot.api")

REQUESTS = Counter("bot_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("bot_request_seconds", "Request latency", ["path"])
REPLIES = Counter("bot_replies_total", "Chat replies", ["mode"])
ESCALATIONS = Counter("bot_escalations_total", "Human handoffs", ["reason"])

limiter = Limiter(key_func=get_remote_address)
_fb_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.retriever = Retriever.from_file(s.kb_path)
    app.state.sessions = SessionStore(s.session_ttl_s, s.max_sessions, s.max_history_turns)
    app.state.llm = LLMAnswerer(s.anthropic_api_key, s.llm_model, s.llm_timeout_s, s.llm_max_tokens) if s.anthropic_api_key else None
    if not s.api_key_set:
        log.warning("BOT_API_KEYS is empty: /v1/* authentication is DISABLED. Set it in production.")
    log.info("ready: %d KB entries, llm=%s", len(app.state.retriever.entries), s.llm_model if app.state.llm else "off")
    yield


app = FastAPI(title="Customer Support Chatbot", version="2.0.0", lifespan=lifespan,
              description="Answers customer questions from a knowledge base and hands off to humans when needed.")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
if settings.cors_origin_list:
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    request.state.request_id = rid
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"request_id": rid, "path": request.url.path})
        response = JSONResponse({"detail": "Internal server error", "request_id": rid}, status_code=500)
    elapsed = time.perf_counter() - start
    route = request.scope.get("route")
    path = route.path if route else "unmatched"
    REQUESTS.labels(path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    response.headers.update({"X-Request-ID": rid, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"})
    log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code, "ms": round(elapsed * 1000, 1)})
    return response


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header), s: Settings = Depends(get_settings)) -> None:
    allowed = s.api_key_set
    if not allowed:
        return
    if not key or not any(secrets.compare_digest(key, k) for k in allowed):
        raise HTTPException(status_code=401, detail="Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


HANDOFF_REPLY = "I'm sorry for the trouble. I've passed this to our support team and a person will follow up. You can add any details here that will help them."
FALLBACK_REPLY = "I couldn't find a reliable answer to that. Try rephrasing, or type 'agent' and I'll connect you with a person."


def _handle_chat(request: Request, body: ChatRequest, background: BackgroundTasks, s: Settings) -> ChatResponse:
    rid = request.state.request_id
    if len(body.message) > s.max_message_chars:
        raise HTTPException(status_code=413, detail=f"Message exceeds {s.max_message_chars} characters.")
    store: SessionStore = request.app.state.sessions
    retriever: Retriever = request.app.state.retriever
    llm: LLMAnswerer | None = request.app.state.llm

    sid = body.session_id or uuid.uuid4().hex
    session = store.get(sid)
    triage = assess(body.message, session.negative_turns)
    if triage.sentiment == "negative":
        session.negative_turns += 1
    hits = retriever.search(body.message, s.top_k)
    top = hits[0].score if hits else 0.0
    sources = [Source(id=h.id, question=h.question, score=h.score) for h in hits if h.score >= s.answer_threshold]

    history = list(session.turns)
    if triage.escalate:
        reply, mode, answered = HANDOFF_REPLY, "escalation", False
        # Include a KB answer when we are confident, so the customer still gets help while waiting.
        if top >= s.answer_threshold:
            reply = f"{hits[0].answer}\n\n{HANDOFF_REPLY}"
    elif top >= s.answer_threshold:
        answered = True
        text = llm.answer(body.message, history, [h for h in hits if h.score >= s.answer_threshold]) if llm else None
        reply, mode = (text, "llm") if text else (hits[0].answer, "knowledge_base")
    else:
        reply, mode, answered = FALLBACK_REPLY, "fallback", False

    store.append(sid, "user", body.message)
    store.append(sid, "assistant", reply)

    escalated = triage.escalate
    if escalated and not session.escalated:
        session.escalated = True
        ESCALATIONS.labels(triage.reason or "unknown").inc()
        payload = {
            "reason": triage.reason,
            "session_id": sid,
            "customer_email": body.customer_email,
            "sentiment": triage.sentiment,
            "last_message": body.message,
            "transcript": [*history, {"role": "user", "content": body.message}, {"role": "assistant", "content": reply}],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        background.add_task(escalation.notify, s.escalation_webhook_url, s.escalation_webhook_secret, payload)

    REPLIES.labels(mode).inc()
    # Customer message text is never logged; only metadata.
    log.info("chat", extra={"request_id": rid, "mode": mode, "escalated": escalated, "reason": triage.reason, "score": top})
    return ChatResponse(reply=reply, session_id=sid, answered=answered, mode=mode, confidence=round(top, 4), sentiment=triage.sentiment, escalated=escalated, sources=sources, request_id=rid)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz(request: Request) -> dict:
    return {"status": "ready", "kb_entries": len(request.app.state.retriever.entries), "llm_enabled": request.app.state.llm is not None, "escalation_configured": bool(get_settings().escalation_webhook_url)}


@app.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_chat)
def chat(request: Request, body: ChatRequest, background: BackgroundTasks, s: Settings = Depends(get_settings)):
    return _handle_chat(request, body, background, s)


@app.post("/demo/chat", response_model=ChatResponse, tags=["demo"])
@limiter.limit(lambda: get_settings().rate_limit_chat)
def demo_chat(request: Request, body: ChatRequest, background: BackgroundTasks, s: Settings = Depends(get_settings)):
    """Open endpoint for the web widget. Disable with BOT_DEMO_ENABLED=false."""
    if not s.demo_enabled:
        raise HTTPException(status_code=404, detail="Demo endpoint disabled.")
    return _handle_chat(request, body, background, s)


@app.post("/v1/feedback", tags=["chat"], dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_chat)
def feedback(request: Request, body: FeedbackRequest, s: Settings = Depends(get_settings)) -> dict:
    s.feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with _fb_lock, s.feedback_path.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **body.model_dump()}) + "\n")
    return {"stored": True}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html", media_type="text/html")
