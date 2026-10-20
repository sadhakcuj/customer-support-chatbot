from unittest.mock import patch


def chat(client, auth, message, **kw):
    return client.post("/v1/chat", json={"message": message, **kw}, headers=auth)


def test_health_ready_ui(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz").json()
    assert r["status"] == "ready" and r["kb_entries"] >= 15 and r["llm_enabled"] is False
    assert "Support Chat" in client.get("/").text


def test_auth_required(client):
    assert client.post("/v1/chat", json={"message": "hi there"}).status_code == 401
    assert client.get("/metrics").status_code == 401


def test_answers_from_knowledge_base(client, auth):
    r = chat(client, auth, "What are your business hours?").json()
    assert r["answered"] and r["mode"] == "knowledge_base" and r["sources"][0]["id"] == "hours"
    assert "Monday to Friday" in r["reply"] and r["escalated"] is False


def test_paraphrase(client, auth):
    r = chat(client, auth, "where is my parcel right now").json()
    assert r["sources"][0]["id"] == "tracking"


def test_out_of_scope_falls_back(client, auth):
    r = chat(client, auth, "what is the capital of France").json()
    assert r["answered"] is False and r["mode"] == "fallback" and r["sources"] == []


def test_session_roundtrip_and_validation(client, auth):
    first = chat(client, auth, "do you offer discounts").json()
    again = chat(client, auth, "and shipping?", session_id=first["session_id"]).json()
    assert again["session_id"] == first["session_id"]
    assert chat(client, auth, "   ").status_code == 422
    assert chat(client, auth, "hello", session_id="bad id!").status_code == 422
    assert chat(client, auth, "x" * 1500).status_code == 413


def test_human_request_escalates_and_notifies_once(client, auth):
    with patch("app.main.escalation.notify") as notify:
        a = chat(client, auth, "I want to talk to a real person").json()
        b = chat(client, auth, "agent please", session_id=a["session_id"]).json()
    assert a["escalated"] and a["mode"] == "escalation"
    assert b["escalated"]
    assert notify.call_count == 1  # one ticket per conversation, not one per message
    payload = notify.call_args.args[2]
    assert payload["reason"] == "customer_asked_for_human" and payload["transcript"]


def test_sustained_negative_sentiment_escalates(client, auth):
    with patch("app.main.escalation.notify") as notify:
        a = chat(client, auth, "this is terrible service").json()
        assert not a["escalated"] and a["sentiment"] == "negative"
        b = chat(client, auth, "you are useless, I am so frustrated", session_id=a["session_id"]).json()
    assert b["escalated"] and notify.call_count == 1


def test_legal_threat_escalates_immediately(client, auth):
    with patch("app.main.escalation.notify"):
        r = chat(client, auth, "I will start a chargeback and call my lawyer").json()
    assert r["escalated"]


def test_demo_endpoint_rate_limited(client):
    codes = [client.post("/demo/chat", json={"message": "hours?"}).status_code for _ in range(8)]
    assert codes[:6] == [200] * 6 and 429 in codes[6:]


def test_feedback(client, auth):
    sid = "abcdefgh12345678"
    assert client.post("/v1/feedback", json={"session_id": sid, "helpful": True}, headers=auth).json() == {"stored": True}


def test_message_text_not_logged(client, auth, caplog):
    import logging
    with caplog.at_level(logging.INFO):
        chat(client, auth, "my secret order number is 99887766")
    assert "99887766" not in caplog.text
