"""Runtime configuration from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOT_", env_file=".env", extra="ignore")

    api_keys: str = ""  # comma-separated; empty disables auth on /v1/* (dev only)
    kb_path: Path = ROOT / "data" / "knowledge_base.json"
    feedback_path: Path = ROOT / "data" / "feedback.jsonl"

    # Retrieval: cosine similarity cut-offs, measured on eval/eval_set.json (see README).
    answer_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    top_k: int = Field(default=3, ge=1, le=10)

    # Optional LLM (Anthropic). Without a key the bot answers from the knowledge base verbatim.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    llm_timeout_s: float = 15.0
    llm_max_tokens: int = 400

    # Human handoff: POSTed as JSON to an n8n webhook. Empty = log only.
    escalation_webhook_url: str = ""
    escalation_webhook_secret: str = ""

    session_ttl_s: int = 1800
    max_sessions: int = 5000
    max_history_turns: int = 6
    max_message_chars: int = 1000

    rate_limit_chat: str = "30/minute"
    demo_enabled: bool = True
    cors_origins: str = ""
    log_level: str = "INFO"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
