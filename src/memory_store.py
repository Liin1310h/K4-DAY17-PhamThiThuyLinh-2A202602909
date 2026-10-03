from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: ~4 characters per token."""
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore — persistent User.md per user
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = "# User Profile\n\n(no information yet)\n"

_SLUG_RE = re.compile(r"[^\w\-]")


def _slugify(user_id: str) -> str:
    return _SLUG_RE.sub("_", user_id).lower()


@dataclass
class UserProfileStore:
    """Persistent storage for User.md, one file per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        user_dir = self.root_dir / _slugify(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return _DEFAULT_PROFILE
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if not p.exists():
            return 0
        return p.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse key: value lines from User.md into a dict."""
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if ":" in line and not line.startswith("#"):
                key, _, value = line.partition(":")
                k = key.strip().lower()
                v = value.strip()
                if k and v:
                    result[k] = v
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a fact line in User.md."""
        current = self.read_text(user_id)
        pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE | re.IGNORECASE)
        new_line = f"{key}: {value}"
        if pattern.search(current):
            updated = pattern.sub(new_line, current)
        else:
            if current.strip() == _DEFAULT_PROFILE.strip():
                updated = f"# User Profile\n\n{new_line}\n"
            else:
                updated = current.rstrip() + f"\n{new_line}\n"
        self.write_text(user_id, updated)


# ---------------------------------------------------------------------------
# extract_profile_updates — regex-based fact extractor
# ---------------------------------------------------------------------------

# Confidence: only facts explicitly stated, not questions or noise.
_QUESTION_INDICATORS = re.compile(
    r"\b(có phải|phải không|không biết|hỏi|sao|thế nào|bạn nghĩ|bạn có|mình có)\b",
    re.IGNORECASE,
)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tên", re.compile(
        # Require name to start with an actual uppercase letter (no IGNORECASE)
        r"(?:[Tt]ên\s+(?:[Mm]ình|[Tt]ôi|là)\s+|[Tt]ôi\s+[Tt]ên\s+|[Mm]ình\s+[Tt]ên\s+|"
        r"[Tt]ên\s+là\s+|my name is\s+|[Mm]y name is\s+)"
        r"([A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)*)",
        # No IGNORECASE — names must start with actual uppercase
    )),
    ("nơi ở", re.compile(
        r"(?:(?:đang\s+)?(?:sống|ở|live|ở tại|sinh sống|hiện tại ở|hiện ở)\s+(?:tại\s+|ở\s+)?)"
        r"([A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zA-ZÀ-ỹ][\w\sÀ-ỹ]{1,30})",
        re.IGNORECASE,
    )),
    ("nghề nghiệp", re.compile(
        r"(?:(?:làm\s+(?:nghề\s+)?|work as\s+|nghề\s+(?:của\s+mình\s+)?là\s+|"
        r"chuyên\s+(?:về|ngành)\s+|là\s+(?:kỹ sư|lập trình|developer|designer|"
        r"giáo viên|bác sĩ|kế toán|marketing|engineer|manager|analyst)))"
        r"([\w\sÀ-ỹ]{2,30}?)(?:\.|,|$|\s+(?:và|nhưng|ở|tại))",
        re.IGNORECASE,
    )),
    ("đồ uống yêu thích", re.compile(
        r"(?:(?:thích|yêu thích|hay uống|thường uống)\s+(?:uống\s+)?)"
        r"([\w\sÀ-ỹ]{2,30}?)(?:\.|,|$|\s+(?:và|nhưng))",
        re.IGNORECASE,
    )),
    ("sở thích", re.compile(
        r"(?:(?:thích|đam mê|quan tâm đến|hứng thú với|sở thích là)\s+)"
        r"([\w\sÀ-ỹ]{2,40}?)(?:\.|,|$|\s+(?:và|nhưng|nên|vì))",
        re.IGNORECASE,
    )),
    ("phong cách trả lời", re.compile(
        r"(?:(?:trả lời|reply|respond)\s+(?:theo\s+kiểu|bằng|dạng)\s+)"
        r"([\w\sÀ-ỹ]{2,30}?)(?:\.|,|$)",
        re.IGNORECASE,
    )),
]

# Noise phrases that should NOT be extracted as facts
_NOISE_PHRASES = {
    "product manager",  # joke/context, not real profession
}

# Common Vietnamese function words that must not be captured as names/values
_FUNCTION_WORDS = {"là", "gì", "có", "không", "và", "ở", "tại", "trong", "ngoài"}

# Base confidence score per fact key — how reliable the pattern is
_BASE_CONFIDENCE: dict[str, float] = {
    "tên": 0.90,             # "tên là X" is very explicit
    "nơi ở": 0.85,           # "đang sống ở X" is fairly explicit
    "nghề nghiệp": 0.80,     # profession patterns are reasonably specific
    "đồ uống yêu thích": 0.85,
    "sở thích": 0.70,        # interest patterns can be noisy
    "phong cách trả lời": 0.80,
}

# Phrases that signal uncertainty → multiply confidence by 0.4
_HEDGING_RE = re.compile(
    r"\b(có lẽ|hình như|không chắc|nghĩ là|có thể|dường như|chắc là|"
    r"maybe|probably|perhaps|not sure)\b",
    re.IGNORECASE,
)

# Phrases that signal temporary/context-specific states → multiply by 0.5
_TEMPORARY_RE = re.compile(
    r"\b(hôm nay|hôm qua|tuần này|tháng này|tạm thời|lần này|chuyến này|"
    r"today|this week|temporarily|for now)\b",
    re.IGNORECASE,
)


def extract_profile_updates(message: str) -> dict[str, tuple[str, float]]:
    """Extract stable profile facts with confidence scores.

    Returns: dict mapping fact_key → (value, confidence) where
    confidence is in [0.0, 1.0].

    Facts with confidence below the caller's threshold should NOT be
    written to User.md.
    """
    # Skip questions — they don't provide new facts
    if "?" in message:
        return {}
    if _QUESTION_INDICATORS.search(message):
        return {}

    # Compute message-level confidence multipliers
    multiplier = 1.0
    if _HEDGING_RE.search(message):
        multiplier *= 0.4
    if _TEMPORARY_RE.search(message):
        multiplier *= 0.5

    results: dict[str, tuple[str, float]] = {}
    for key, pattern in _PATTERNS:
        m = pattern.search(message)
        if m:
            value = m.group(1).strip().rstrip(".,;")
            value_lower = value.lower().strip()
            if (
                value_lower not in _NOISE_PHRASES
                and value_lower not in _FUNCTION_WORDS
                and len(value) >= 2
            ):
                if key == "tên" and not any(c.isupper() for c in value):
                    continue
                base = _BASE_CONFIDENCE.get(key, 0.75)
                confidence = round(base * multiplier, 3)
                results[key] = (value, confidence)
    return results


# ---------------------------------------------------------------------------
# summarize_messages — heuristic summary
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact bullet-point summary of older messages."""
    if not messages:
        return ""
    selected = messages[:max_items]
    lines: list[str] = ["[Tóm tắt hội thoại cũ]"]
    for msg in selected:
        role = msg.get("role", "?")
        content = msg.get("content", "").strip()
        # Truncate long content
        if len(content) > 120:
            content = content[:120] + "…"
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CompactMemoryManager
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the N most-recent messages in full.
    When total token count exceeds threshold, older messages are compacted
    into a rolling summary.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]  # type: ignore[return-value]

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and trigger compaction if threshold exceeded."""
        ts = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = ts["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> None:
        ts = self.state[thread_id]
        messages: list[dict[str, str]] = ts["messages"]  # type: ignore
        keep = self.keep_messages
        to_summarize = messages[:-keep] if len(messages) > keep else []
        recent = messages[-keep:]

        if to_summarize:
            new_summary = summarize_messages(to_summarize, max_items=len(to_summarize))
            existing = ts["summary"]
            if existing:
                ts["summary"] = existing + "\n\n" + new_summary
            else:
                ts["summary"] = new_summary
            ts["messages"] = recent
            ts["compactions"] = int(ts["compactions"]) + 1  # type: ignore

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with messages, summary, compactions."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        ts = self._ensure_thread(thread_id)
        return int(ts["compactions"])  # type: ignore
