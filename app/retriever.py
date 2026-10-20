"""Knowledge-base retrieval: TF-IDF (word + character n-grams) with cosine similarity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack


# Domain concept groups. When any phrase of a group appears, a concept tag is appended to the text so that
# paraphrases ("parcel", "shipment") meet on a shared token. Deterministic, offline, and easy to extend.
CONCEPTS: dict[str, list[str]] = {
    "package": ["parcel", "shipment", "package", "box"],
    "refund": ["refund", "money back", "reimburse", "get my money", "money is back"],
    "return": ["return", "send back", "sending back", "sent back", "exchange"],
    "damaged": ["broken", "smashed", "cracked", "damaged", "defective", "faulty", "crushed", "wrong item", "wrong colour", "wrong color", "wrong size", "not working", "stopped working", "doesn't work"],
    "international": ["abroad", "overseas", "international", "worldwide", "other countries", "outside the us", "uk", "canada", "australia", "india", "europe", "germany", "mexico", "japan"],
    "delete": ["erase", "delete", "close my account", "remove my", "permanently"],
    "invoice": ["receipt", "invoice", "proof of purchase", "tax", "billing statement"],
    "discount": ["coupon", "promo", "promotion", "sale", "sales", "deal", "discount", "offer", "loyalty", "black friday"],
    "warranty": ["guarantee", "warranty", "covered", "coverage"],
    "contact": ["phone", "call you", "email", "helpline", "customer care", "customer service", "reach you", "speak with", "who do i", "get help", "support team"],
    "payment": ["amex", "visa", "mastercard", "google pay", "paypal", "credit card", "apple pay", "pay with", "pay using", "payment"],
    "login": ["login", "log in", "sign in", "credentials", "password", "locked", "get back into"],
    "hours": ["weekend", "saturday", "sunday", "opening", "closing", "close on", "hours", "what time", "available on", "working on", "working hours"],
    "tracking": ["track", "tracking", "where is my", "where is the", "follow my", "status", "has it shipped"],
    "bulk": ["wholesale", "bulk", "large order", "units", "company", "business", "volume", "in quantity"],
    "app": ["app", "iphone", "android", "play store", "app store", "ios"],
    "cancel": ["cancel", "stop my order", "stop it", "changed my mind", "ordered by mistake", "undo"],
    "address": ["address", "street", "ships to", "moved house", "where it ships", "deliver to a different"],
    "shipping_cost": ["shipping cost", "delivery charge", "delivery cost", "postage", "free delivery", "free shipping", "minimum spend", "spend to avoid"],
    "speed": ["how soon", "how long", "when will", "when should i expect", "fastest", "expect my", "how many days will"],
    "security": ["secure", "safe", "encrypt", "privacy", "protect"],
    "secret": [],
}
_CONCEPT_RES = {tag: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)", re.I) for tag, phrases in CONCEPTS.items() if phrases}


def expand(text: str) -> str:
    """Append concept tags for any group present in ``text``."""
    tags = [f"zz{tag}" for tag, rx in _CONCEPT_RES.items() if rx.search(text)]
    return f"{text} {' '.join(tags)}" if tags else text


@dataclass(frozen=True)
class Hit:
    id: str
    question: str
    answer: str
    score: float


class Retriever:
    def __init__(self, entries: list[dict]) -> None:
        if not entries:
            raise ValueError("knowledge base is empty")
        ids = [e["id"] for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError("knowledge base has duplicate ids")
        self.entries = entries
        # One document per entry: question plus aliases (paraphrases), so matching is semantic-ish
        # without a model download. Answers are excluded to avoid matching on answer wording.
        docs = [expand(" ".join([e["question"], *e.get("aliases", [])])) for e in entries]
        self._word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", strip_accents="unicode")
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._matrix = hstack([self._word.fit_transform(docs), self._char.fit_transform(docs)]).tocsr()
        self._norm = np.sqrt(np.asarray(self._matrix.multiply(self._matrix).sum(axis=1))).ravel()

    @classmethod
    def from_file(cls, path: Path) -> "Retriever":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        query = expand(query)
        q = hstack([self._word.transform([query]), self._char.transform([query])]).tocsr()
        qn = float(np.sqrt(q.multiply(q).sum()))
        if qn == 0.0:
            return []
        sims = linear_kernel(q, self._matrix).ravel() / (qn * np.where(self._norm == 0, 1, self._norm))
        order = np.argsort(-sims)[:k]
        return [
            Hit(self.entries[i]["id"], self.entries[i]["question"], self.entries[i]["answer"], round(float(sims[i]), 4))
            for i in order
        ]
