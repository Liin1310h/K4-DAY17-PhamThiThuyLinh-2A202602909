from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A — within-session memory only.

    - Remembers messages within the same thread_id.
    - No persistent User.md.
    - Forgets everything when a new thread starts.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return agent response and token accounting."""
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative agent-generated token count for one thread."""
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt context tokens processed for one thread."""
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _get_or_create_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline path.

        Stores the message in session history, generates a reply that can
        answer questions about facts mentioned WITHIN this thread only.
        Never remembers facts from other thread IDs.
        """
        sess = self._get_or_create_session(thread_id)

        # Estimate prompt cost: all previous messages + current message
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in sess.messages)
        prompt_tokens += estimate_tokens(message)
        sess.prompt_tokens_processed += prompt_tokens

        # Store user message
        sess.messages.append({"role": "user", "content": message})

        # Generate deterministic response based on in-thread history
        response_text = self._generate_offline_response(sess.messages, message)

        # Estimate agent output tokens
        output_tokens = estimate_tokens(response_text)
        sess.token_usage += output_tokens

        # Store assistant reply
        sess.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_offline_response(
        self, history: list[dict[str, str]], current_message: str
    ) -> str:
        """Build a response using only in-thread history."""
        # Collect facts from this thread only
        thread_facts: dict[str, str] = {}
        for msg in history[:-1]:  # exclude the just-appended current message
            if msg["role"] == "user":
                self._extract_simple_facts(msg["content"], thread_facts)

        msg_lower = current_message.lower()

        # Answer recall-style questions from in-thread facts
        if any(kw in msg_lower for kw in ["tên", "name"]):
            if "tên" in thread_facts:
                return f"Trong cuộc hội thoại này, bạn đã đề cập tên là {thread_facts['tên']}."
            return "Trong hội thoại này bạn chưa cho tôi biết tên."

        if any(kw in msg_lower for kw in ["ở đâu", "sống ở", "nơi ở", "location", "city"]):
            if "nơi ở" in thread_facts:
                return f"Bạn đã nhắc đến bạn đang ở {thread_facts['nơi ở']}."
            return "Trong hội thoại này bạn chưa cho tôi biết nơi ở."

        if any(kw in msg_lower for kw in ["nghề", "làm gì", "profession", "work"]):
            if "nghề nghiệp" in thread_facts:
                return f"Bạn đã nhắc đến nghề nghiệp của bạn là {thread_facts['nghề nghiệp']}."
            return "Trong hội thoại này bạn chưa đề cập nghề nghiệp."

        if any(kw in msg_lower for kw in ["uống", "thích uống", "drink"]):
            if "đồ uống yêu thích" in thread_facts:
                return f"Bạn thích uống {thread_facts['đồ uống yêu thích']}."
            return "Tôi không biết bạn thích uống gì trong hội thoại này."

        if any(kw in msg_lower for kw in ["nhắc lại", "nhớ", "recall", "remember"]):
            if thread_facts:
                lines = [f"- {k}: {v}" for k, v in thread_facts.items()]
                return "Những gì bạn đã chia sẻ trong cuộc hội thoại này:\n" + "\n".join(lines)
            return "Bạn chưa chia sẻ thông tin cá nhân trong hội thoại này."

        # Generic echo response
        return f"Tôi đã ghi nhận: {current_message[:80]}{'…' if len(current_message) > 80 else ''}. Tôi chỉ nhớ trong phiên hội thoại hiện tại."

    @staticmethod
    def _extract_simple_facts(text: str, facts: dict[str, str]) -> None:
        """Quick in-thread fact extractor (no cross-session persistence)."""
        import re

        name_m = re.search(
            r"(?:tên\s+(?:mình|tôi|là)\s+|tôi\s+tên\s+|mình\s+tên\s+|tên\s+là\s+)"
            r"([A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zA-ZÀ-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zA-ZÀ-ỹ]+)*)",
            text, re.IGNORECASE,
        )
        if name_m:
            facts["tên"] = name_m.group(1).strip()

        loc_m = re.search(
            r"(?:đang\s+)?(?:sống|ở|ở tại)\s+(?:tại\s+|ở\s+)?"
            r"([A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ][a-zA-ZÀ-ỹ]+)",
            text, re.IGNORECASE,
        )
        if loc_m:
            facts["nơi ở"] = loc_m.group(1).strip()

        job_m = re.search(
            r"(?:làm\s+)([a-zA-ZÀ-ỹ\s]{3,25})(?:\.|,|$| và| nhưng| ở)",
            text, re.IGNORECASE,
        )
        if job_m:
            facts["nghề nghiệp"] = job_m.group(1).strip()

        drink_m = re.search(
            r"(?:thích\s+uống\s+|hay\s+uống\s+)([\w\sÀ-ỹ]{2,20}?)(?:\.|,|$)",
            text, re.IGNORECASE,
        )
        if drink_m:
            facts["đồ uống yêu thích"] = drink_m.group(1).strip()

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Live path using LangChain agent (if available)."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.langchain_agent.invoke(
                {"messages": [("human", message)]}, config=config
            )
            response_text = result["messages"][-1].content
            output_tokens = estimate_tokens(response_text)
            sess = self._get_or_create_session(thread_id)
            sess.token_usage += output_tokens
            prompt_tokens = estimate_tokens(message)
            sess.prompt_tokens_processed += prompt_tokens
            return {
                "response": response_text,
                "agent_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
            }
        except Exception:
            return self._reply_offline(thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        """Optionally wire LangChain/LangGraph InMemorySaver agent."""
        try:
            from langchain_core.messages import SystemMessage
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent
            from model_provider import build_chat_model

            llm = build_chat_model(self.config.model)
            if llm is None:
                return

            system_prompt = (
                "You are a helpful assistant. You only remember what happened "
                "in the current conversation thread. You do not have any "
                "long-term memory across different sessions."
            )
            checkpointer = MemorySaver()
            self.langchain_agent = create_react_agent(
                llm, tools=[], checkpointer=checkpointer,
                state_modifier=system_prompt,
            )
        except Exception:
            self.langchain_agent = None
