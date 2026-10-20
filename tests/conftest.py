import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["BOT_API_KEYS"] = "test-key"
os.environ["BOT_FEEDBACK_PATH"] = str(Path(_tmp) / "fb.jsonl")
os.environ["BOT_RATE_LIMIT_CHAT"] = "6/minute"
os.environ["BOT_ESCALATION_WEBHOOK_URL"] = ""
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest
from fastapi.testclient import TestClient

from app.main import app, limiter


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset():
    limiter.reset()
    yield


@pytest.fixture
def auth():
    return {"X-API-Key": "test-key"}
