from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    customer_email: str | None = Field(default=None, max_length=254)

    @field_validator("message")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message is empty")
        return v


class Source(BaseModel):
    id: str
    question: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    answered: bool = Field(description="False when the bot could not find a grounded answer.")
    mode: Literal["llm", "knowledge_base", "fallback", "escalation"]
    confidence: float = Field(description="Top retrieval similarity in [0, 1]. A retrieval score, not a calibrated probability.")
    sentiment: Literal["negative", "neutral", "positive"]
    escalated: bool
    sources: list[Source]
    request_id: str


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    helpful: bool
    comment: str = Field(default="", max_length=1000)
