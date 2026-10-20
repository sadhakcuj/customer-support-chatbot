"""Bounded in-memory conversation store with TTL. Single-process; use Redis for multi-instance scale."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    turns: list[dict] = field(default_factory=list)  # {"role": "user"|"assistant", "content": str}
    negative_turns: int = 0
    escalated: bool = False
    touched: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_s: int, max_sessions: int, max_turns: int) -> None:
        self._ttl, self._max, self._max_turns = ttl_s, max_sessions, max_turns
        self._data: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, sid: str) -> Session:
        now = time.time()
        with self._lock:
            for key in [k for k, v in self._data.items() if now - v.touched > self._ttl]:
                del self._data[key]
            s = self._data.pop(sid, None) or Session()
            s.touched = now
            self._data[sid] = s
            while len(self._data) > self._max:
                self._data.popitem(last=False)
            return s

    def append(self, sid: str, role: str, content: str) -> None:
        with self._lock:
            s = self._data.get(sid)
            if s is None:
                return
            s.turns.append({"role": role, "content": content})
            s.turns = s.turns[-self._max_turns * 2 :]

    def __len__(self) -> int:
        return len(self._data)
