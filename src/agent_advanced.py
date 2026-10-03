from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B — three-layer memory: short-term + User.md + compact.

    Pipeline per turn:
    1. Extract stable facts from incoming message → persist to User.md
    2. Append message to CompactMemoryManager (auto-compacts when needed)
    3. Build prompt = User.md + compact summary + recent messages
    4. Generate response → update token counters
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline and live mode."""
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path with all three memory layers."""

        # 1. Extract facts with confidence scores; persist only above threshold
        extracted = extract_profile_updates(message)
        for key, (value, confidence) in extracted.items():
            if confidence >= self.config.confidence_threshold:
                self._upsert_profile_fact(user_id, key, value)

        # 2. Append to compact memory (may trigger compaction)
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt-context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate deterministic response
        response_text = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply and update counters
        self.compact_memory.append(thread_id, "assistant", response_text)
        output_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + output_tokens
        )

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _upsert_profile_fact(self, user_id: str, key: str, value: str) -> None:
        """Confidence-aware upsert: update User.md only for non-empty values."""
        if not value or len(value) < 2:
            return
        self.profile_store.upsert_fact(user_id, key, value)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the total context carried into one turn.

        Includes: User.md + compact summary + recent kept messages.
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text: str = ctx.get("summary", "")  # type: ignore
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore

        recent_text = " ".join(m["content"] for m in messages)
        total = (
            estimate_tokens(profile_text)
            + estimate_tokens(summary_text)
            + estimate_tokens(recent_text)
        )
        return max(1, total)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Answers recall questions about facts stored in User.md and
        in-thread compact memory.
        """
        profile_text = self.profile_store.read_text(user_id)
        facts = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")  # type: ignore
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore

        msg_lower = message.lower()

        # -- Name recall
        if any(kw in msg_lower for kw in ["tên", "name"]):
            val = facts.get("tên") or facts.get("ten")
            if val:
                return f"Bạn tên là {val}."
            return "Tôi chưa lưu tên của bạn."

        # -- Location recall
        if any(kw in msg_lower for kw in ["ở đâu", "sống ở", "nơi ở", "location", "thành phố", "city", "đang ở"]):
            val = facts.get("nơi ở") or facts.get("noi o")
            if val:
                return f"Bạn đang sống ở {val}."
            return "Tôi chưa lưu nơi ở của bạn."

        # -- Profession recall
        if any(kw in msg_lower for kw in ["nghề", "làm gì", "profession", "work", "job", "công việc"]):
            val = facts.get("nghề nghiệp") or facts.get("nghe nghiep")
            if val:
                return f"Nghề nghiệp của bạn là {val}."
            return "Tôi chưa lưu nghề nghiệp của bạn."

        # -- Drink preference
        if any(kw in msg_lower for kw in ["uống", "drink", "thức uống"]):
            val = facts.get("đồ uống yêu thích") or facts.get("do uong")
            if val:
                return f"Bạn thích uống {val}."
            return "Tôi chưa lưu sở thích đồ uống của bạn."

        # -- Interests / hobbies
        if any(kw in msg_lower for kw in ["sở thích", "thích gì", "interest", "hobby"]):
            val = facts.get("sở thích") or facts.get("so thich")
            if val:
                return f"Sở thích của bạn là {val}."
            return "Tôi chưa lưu sở thích của bạn."

        # -- Reply style
        if any(kw in msg_lower for kw in ["phong cách", "style", "cách trả lời"]):
            val = facts.get("phong cách trả lời")
            if val:
                return f"Bạn thích được trả lời theo phong cách: {val}."
            return "Tôi chưa lưu phong cách trả lời của bạn."

        # -- Full recall
        if any(kw in msg_lower for kw in ["nhắc lại", "nhớ gì", "recall", "remember", "biết gì về", "thông tin"]):
            if facts and not _only_default(profile_text):
                lines = [f"- {k}: {v}" for k, v in facts.items()]
                return "Những gì tôi biết về bạn:\n" + "\n".join(lines)
            return "Tôi chưa lưu thông tin gì về bạn."

        # -- Generic: acknowledge and mention any facts stored
        fact_hint = ""
        if facts and not _only_default(profile_text):
            names = list(facts.keys())[:2]
            fact_hint = f" (Tôi đã lưu: {', '.join(names)} của bạn.)"

        return (
            f"Tôi đã nhận được: {message[:80]}{'…' if len(message) > 80 else ''}."
            f"{fact_hint}"
        )

    # ------------------------------------------------------------------
    # Live path
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path — extract facts first, then call LangChain agent."""
        extracted = extract_profile_updates(message)
        for key, (value, confidence) in extracted.items():
            if confidence >= self.config.confidence_threshold:
                self._upsert_profile_fact(user_id, key, value)

        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [("human", message)]}, config=config
            )
            response_text = result["messages"][-1].content
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", response_text)

        output_tokens = estimate_tokens(response_text)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        """Wire a live agent with User.md tools and compact middleware."""
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from model_provider import build_chat_model

            llm = build_chat_model(self.config.model)
            if llm is None:
                return

            profile_store = self.profile_store

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persistent User.md for a given user."""
                return profile_store.read_text(user_id)

            @tool
            def write_user_fact(user_id: str, key: str, value: str) -> str:
                """Upsert a fact into the user's persistent profile."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Saved {key}: {value}"

            system_prompt = (
                "You are a helpful assistant with long-term memory. "
                "Use read_user_profile to recall facts about the user. "
                "Use write_user_fact to save new facts the user tells you. "
                "Always personalize your responses using the user's stored profile."
            )
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm,
                tools=[read_user_profile, write_user_fact],
                checkpointer=checkpointer,
                state_modifier=system_prompt,
            )
        except Exception:
            self.langchain_agent = None


def _only_default(profile_text: str) -> bool:
    return "(no information yet)" in profile_text
