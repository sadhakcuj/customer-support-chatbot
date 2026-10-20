"""Sentiment and escalation signals. Deliberately transparent: lexicon rules, not a black box."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NEGATIVE = {
    "angry", "furious", "terrible", "awful", "horrible", "worst", "useless", "ridiculous", "unacceptable",
    "disgusted", "disappointed", "frustrated", "annoyed", "scam", "fraud", "rip-off", "ripoff", "waste",
    "broken", "never again", "fed up", "sick of", "pathetic", "incompetent",
}
_POSITIVE = {"thanks", "thank you", "great", "perfect", "awesome", "love", "appreciate", "helpful", "excellent"}
_HANDOFF = re.compile(
    r"\b(human|real person|agent|representative|supervisor|manager|speak to someone|talk to someone|call me)\b", re.I
)
_LEGAL = re.compile(r"\b(lawyer|lawsuit|sue|legal action|chargeback|dispute the charge|report you|consumer court)\b", re.I)
_NEGATION = re.compile(r"\b(not|no|never|isn't|wasn't|don't|doesn't)\s+(?:\w+\s+){0,1}$", re.I)


@dataclass(frozen=True)
class Triage:
    sentiment: str  # negative | neutral | positive
    escalate: bool
    reason: str | None


def _count(text: str, words: set[str]) -> int:
    n = 0
    for w in words:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text):
            if not _NEGATION.search(text[: m.start()]):
                n += 1
    return n


def assess(message: str, prior_negative_turns: int = 0) -> Triage:
    text = message.lower()
    neg, pos = _count(text, _NEGATIVE), _count(text, _POSITIVE)
    shouting = sum(c.isupper() for c in message) > 0.6 * max(1, sum(c.isalpha() for c in message)) and len(message) > 15
    if shouting:
        neg += 1
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"

    if _LEGAL.search(message):
        return Triage(sentiment, True, "legal_or_chargeback_threat")
    if _HANDOFF.search(message):
        return Triage(sentiment, True, "customer_asked_for_human")
    if sentiment == "negative" and (neg >= 2 or prior_negative_turns >= 1):
        return Triage(sentiment, True, "sustained_negative_sentiment")
    return Triage(sentiment, False, None)
