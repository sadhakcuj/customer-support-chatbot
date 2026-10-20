# Customer Support Chatbot

A deployable support assistant: it answers from your knowledge base, optionally phrases answers with an LLM that is only allowed to use that knowledge, detects frustrated or legal-risk customers, and hands the conversation to a human through an n8n workflow (ticket row, Slack alert, acknowledgement email).

## What changed from v1, and why

The first version labelled itself "fine-tuned DistilBERT, 85% accuracy", but the model was trained on dummy labels and the 85% was a constant in the code. It has been replaced with a design whose behaviour can be measured:

- Retrieval over a knowledge base with a **measured** accuracy (below), not a claimed one.
- An LLM is optional and **grounded**: it must answer from retrieved entries or abstain, and customer text is treated as untrusted input.
- Without an API key the bot still works, answering from the knowledge base verbatim.
- Escalation is a first-class feature, with a transparent rule set.

## Measured retrieval quality

`python -m eval.run_eval` (results in `eval/results.json`). The answer threshold was chosen on a dev split and reported on a separate test split. Out-of-scope questions (chit-chat, unrelated topics) are included so the bot is also scored on knowing when not to answer.

| Held-out test split | Result |
|---|---|
| In-scope questions answered correctly (top-1) | 94.7% (36/38) |
| Correct entry within top 3 | 97.4% |
| Out-of-scope questions correctly declined | 87.5% (7/8) |

Read these numbers with care: 46 questions is a small set, and the concept-synonym list in `app/retriever.py` was extended *after* a first look at test results, so the test figure is optimistic. The right next step for a real deployment is to evaluate on your own logged customer questions. With the LLM enabled, the model also sees the top 3 entries and may abstain, which covers misses where the right entry is in the top 3.

## How a message is handled

```
message -> triage (sentiment, human request, legal threat)
        -> retrieve top-3 KB entries (TF-IDF + concept tags, cosine similarity)
        -> if similarity >= threshold: LLM grounded answer (or KB answer verbatim)
           else: "I couldn't find that" + offer a human
        -> if escalation: POST transcript to n8n (once per conversation)
```

Escalation triggers: customer asks for a person, mentions a lawyer or chargeback, or negative sentiment repeats across turns. Rules are in `app/triage.py` and unit-tested.

## Run locally

```bash
pip install -r requirements-dev.txt
pytest -q
BOT_API_KEYS=dev-key uvicorn app.main:app --reload     # UI at http://localhost:8000, docs at /docs
```
```bash
curl -X POST localhost:8000/v1/chat -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
  -d '{"message":"How long does delivery take?"}'
```
Reply fields: `reply`, `mode` (`llm`, `knowledge_base`, `fallback`, `escalation`), `confidence` (retrieval similarity, not a calibrated probability), `sentiment`, `escalated`, `sources`, `session_id`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BOT_API_KEYS` | empty | Comma-separated keys for `/v1/*` and `/metrics`. **Set in production.** |
| `ANTHROPIC_API_KEY` | empty | Enables LLM-phrased answers. Optional. |
| `BOT_LLM_MODEL` | `claude-sonnet-5-5` | Model used when a key is set. |
| `BOT_ESCALATION_WEBHOOK_URL` | empty | n8n webhook for human handoff. |
| `BOT_ESCALATION_WEBHOOK_SECRET` | empty | Sent as `X-Webhook-Secret`. |
| `BOT_ANSWER_THRESHOLD` | `0.12` | Minimum similarity to answer. Re-run the eval if you change the KB. |
| `BOT_RATE_LIMIT_CHAT` | `30/minute` | Per-IP limit. |
| `BOT_DEMO_ENABLED` | `true` | Serve the open `/demo/chat` endpoint used by the web widget. |
| `BOT_CORS_ORIGINS` | empty | Allowed origins if you embed the API in another site. |

## Use your own business content

`data/knowledge_base.json` ships with **sample content for a fictional store** (company email, phone, policies). Replace it with your real FAQs before going live: each entry has `question`, `aliases` (paraphrases customers use) and `answer`. Then run `python -m eval.run_eval` and add your own questions to `eval/eval_set.json`.

## Deploy

**Render:** New + > Blueprint, select this repo (`render.yaml`). Set `ANTHROPIC_API_KEY` and `BOT_ESCALATION_WEBHOOK_URL` when prompted. Free instances sleep when idle.
**Docker:** `docker build -t support-chatbot . && docker run -p 8000:8000 -e BOT_API_KEYS=change-me support-chatbot`

Scaling note: conversation history lives in process memory, so run one worker. For several instances, move `app/sessions.py` to Redis.

## n8n human handoff

Import `n8n/escalation_workflow.json`. Create a Header Auth credential named `X-Webhook-Secret` with the same value as `BOT_ESCALATION_WEBHOOK_SECRET`, set the Google Sheet ID and `SLACK_CHANNEL`, connect Gmail, activate, and paste the production webhook URL into `BOT_ESCALATION_WEBHOOK_URL`. This workflow has not been run against live Google, Slack or Gmail accounts; test it with a dummy conversation first.

## Operations

JSON logs with request IDs; customer message text is never logged. `/healthz`, `/readyz`, Prometheus `/metrics` (requests, latency, reply modes, escalations). CI runs tests, the retrieval eval, and a container smoke test.

MIT licensed.
