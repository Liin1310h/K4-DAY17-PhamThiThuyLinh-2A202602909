from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear in the answer."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    ratio = hits / len(expected)
    if ratio == 0:
        return 0.0
    if ratio < 1.0:
        return 0.5
    return 1.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Rewards: answer length, presence of expected terms, no generic fallback.
    """
    if not answer:
        return 0.0

    score = 0.0

    # Length reward: prefer answers of 10–200 chars
    length = len(answer)
    if length >= 10:
        score += 0.3
    if length >= 30:
        score += 0.2

    # Keyword reward
    answer_lower = answer.lower()
    if expected:
        hits = sum(1 for e in expected if e.lower() in answer_lower)
        score += 0.5 * (hits / len(expected))
    else:
        score += 0.5

    return min(1.0, score)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Steps:
    1. Feed all turns to the agent.
    2. Track agent tokens only + prompt tokens processed.
    3. Ask recall questions in a FRESH thread.
    4. Compute average recall and quality.
    5. Record memory file growth and compaction count.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_growth = 0

    for conv in conversations:
        user_id: str = conv.get("user_id", "unknown")
        conv_id: str = conv.get("id", "conv")
        turns: list[str] = conv.get("turns", [])
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Feed all turns under the main thread
        main_thread = f"{conv_id}-main"
        for turn in turns:
            result = agent.reply(user_id, main_thread, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(main_thread)

        # Ask recall questions in a FRESH thread (cross-session test)
        recall_thread = f"{conv_id}-recall"
        for rq in recall_questions:
            question: str = rq.get("question", "")
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(user_id, recall_thread, question)
            answer: str = result.get("response", "")
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    # Memory growth: sum file sizes for all users seen
    for conv in conversations:
        user_id = conv.get("user_id", "unknown")
        try:
            memory_growth += agent.memory_file_size(user_id)
        except AttributeError:
            pass  # baseline has no memory file

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a tabulated comparison table."""
    try:
        from tabulate import tabulate

        headers = [
            "Agent",
            "Agent tokens only",
            "Prompt tokens processed",
            "Cross-session recall",
            "Response quality",
            "Memory growth (bytes)",
            "Compactions",
        ]
        table = [
            [
                r.agent_name,
                r.agent_tokens_only,
                r.prompt_tokens_processed,
                f"{r.recall_score:.3f}",
                f"{r.response_quality:.3f}",
                r.memory_growth_bytes,
                r.compactions,
            ]
            for r in rows
        ]
        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback plain text
        lines = [
            "Agent | Agent tokens | Prompt tokens | Recall | Quality | Mem bytes | Compactions"
        ]
        for r in rows:
            lines.append(
                f"{r.agent_name} | {r.agent_tokens_only} | {r.prompt_tokens_processed} | "
                f"{r.recall_score:.3f} | {r.response_quality:.3f} | "
                f"{r.memory_growth_bytes} | {r.compactions}"
            )
        return "\n".join(lines)


def main() -> None:
    """Run both benchmark suites and print comparison tables."""
    config = load_config(Path(__file__).resolve().parent.parent)

    data_dir = config.data_dir
    std_path = data_dir / "conversations.json"
    stress_path = data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    # --- Standard Benchmark ---
    print("\n" + "=" * 60)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 60)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(std_rows))

    # --- Long-Context Stress Benchmark ---
    print("\n" + "=" * 60)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 60)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))

    # --- Analysis ---
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    _print_analysis(std_rows, stress_rows)


def _print_analysis(
    std_rows: list[BenchmarkRow], stress_rows: list[BenchmarkRow]
) -> None:
    baseline_std = next(r for r in std_rows if r.agent_name == "Baseline")
    advanced_std = next(r for r in std_rows if r.agent_name == "Advanced")
    baseline_stress = next(r for r in stress_rows if r.agent_name == "Baseline")
    advanced_stress = next(r for r in stress_rows if r.agent_name == "Advanced")

    print("\n── CÂU CHUYỆN CỦA HỆ THỐNG ──────────────────────────────────────")

    print("\n1. Baseline không nhớ dài hạn")
    print(
        f"   Cross-session recall = {baseline_std.recall_score:.3f} (standard), "
        f"{baseline_stress.recall_score:.3f} (stress)"
    )
    print(
        "   Baseline chỉ giữ messages trong cùng thread_id. Sang thread mới, "
        "toàn bộ context bị xóa → recall = 0."
    )

    print("\n2. Advanced thêm User.md → recall tăng")
    print(
        f"   Advanced recall: {advanced_std.recall_score:.3f} (standard), "
        f"{advanced_stress.recall_score:.3f} (stress)"
    )
    print(
        "   Mỗi lượt chat, facts ổn định (tên, nơi ở, nghề…) được trích xuất và "
        "ghi vào User.md. Thread mới vẫn đọc được User.md → nhớ qua session."
    )

    print("\n3. Hội thoại ngắn: Advanced tốn hơn về prompt tokens")
    diff = advanced_std.prompt_tokens_processed - baseline_std.prompt_tokens_processed
    sign = "+" if diff >= 0 else ""
    print(
        f"   Advanced: {advanced_std.prompt_tokens_processed} vs "
        f"Baseline: {baseline_std.prompt_tokens_processed} ({sign}{diff} tokens)"
    )
    print(
        "   User.md được nối vào đầu prompt mỗi lượt. Với hội thoại ngắn, "
        "chi phí này không được bù đắp → Advanced đắt hơn ở standard benchmark."
    )

    print("\n4. Hội thoại rất dài: compact memory kéo chi phí xuống")
    print(
        f"   Compactions — Baseline: {baseline_stress.compactions}, "
        f"Advanced: {advanced_stress.compactions}"
    )
    diff_stress = baseline_stress.prompt_tokens_processed - advanced_stress.prompt_tokens_processed
    print(
        f"   Baseline: {baseline_stress.prompt_tokens_processed} tokens, "
        f"Advanced: {advanced_stress.prompt_tokens_processed} tokens "
        f"(tiết kiệm {diff_stress} tokens)"
    )
    print(
        "   Compact nén messages cũ thành summary → chủ yếu tối ưu prompt_tokens_processed, "
        "không ảnh hưởng agent_tokens_only. Baseline tích luỹ toàn bộ lịch sử."
    )

    print("\n5. Memory file growth và rủi ro")
    print(
        f"   User.md size — standard: {advanced_std.memory_growth_bytes} bytes, "
        f"stress: {advanced_stress.memory_growth_bytes} bytes"
    )
    print(
        "   Rủi ro: file phình to nếu lưu noise, lưu sai fact, hoặc user sửa thông tin "
        "nhiều lần mà agent không xử lý đúng conflict."
    )

    print("\n── BONUS: CONFIDENCE THRESHOLD ──────────────────────────────────")
    print(
        "\n   Vấn đề giải quyết:"
    )
    print(
        "   Không phải mọi câu user nói đều là fact ổn định. Ví dụ:"
    )
    print(
        "   • 'Có lẽ mình sẽ về Hà Nội' → nơi ở chưa chắc chắn (confidence ~0.43)"
    )
    print(
        "   • 'Hôm nay mình ở Đà Nẵng họp' → tạm thời, không phải nơi ở thật (confidence ~0.43)"
    )
    print(
        "   • 'Tôi đang sống ở Huế' → rõ ràng, ổn định (confidence 0.85) → GHI vào User.md"
    )
    print(
        "\n   Cách hoạt động:"
    )
    print(
        "   extract_profile_updates() trả về (value, confidence) cho mỗi fact."
    )
    print(
        "   Nếu confidence < threshold (mặc định 0.7) → KHÔNG ghi vào User.md."
    )
    print(
        "   Hedging words ('có lẽ', 'hình như') → nhân confidence × 0.4."
    )
    print(
        "   Temporary words ('hôm nay', 'tạm thời') → nhân confidence × 0.5."
    )
    print(
        "\n   Cải thiện recall và token cost:"
    )
    print(
        "   • Tránh lưu sai fact → User.md chứa thông tin đáng tin cậy hơn → recall chính xác hơn."
    )
    print(
        "   • File nhỏ hơn → prompt_tokens_processed giảm ở mỗi lượt."
    )
    print(
        "\n   Rủi ro tạo ra:"
    )
    print(
        "   • Threshold quá cao → bỏ sót facts thật sự ổn định → recall giảm."
    )
    print(
        "   • User diễn đạt mơ hồ nhưng fact là thật → bị lọc oan."
    )
    print(
        "   • Cần tune threshold theo domain: hội thoại casual cần threshold thấp hơn."
    )

    print("\n── BONUS: CONFLICT HANDLING ─────────────────────────────────────")
    print(
        "\n   Vấn đề giải quyết:"
    )
    print(
        "   User có thể sửa thông tin: 'Tôi vừa chuyển từ Đà Nẵng về Huế rồi'."
    )
    print(
        "   Nếu User.md đang có 'nơi ở: Đà Nẵng', agent phải cập nhật đúng."
    )
    print(
        "\n   Cách hoạt động:"
    )
    print(
        "   upsert_fact() dùng regex tìm dòng 'nơi ở: ...' trong User.md,"
    )
    print(
        "   thay thế bằng giá trị mới → không giữ cả hai fact cũ và mới."
    )
    print(
        "\n   Cải thiện recall:"
    )
    print(
        "   Khi user hỏi lại 'Mình đang ở đâu?', agent trả lời đúng 'Huế'"
    )
    print(
        "   thay vì 'Đà Nẵng' (fact cũ) hay cả hai (mâu thuẫn)."
    )
    print(
        "\n   Rủi ro tạo ra:"
    )
    print(
        "   • Nếu fact mới bị extract sai, fact đúng cũ bị ghi đè mất."
    )
    print(
        "   • Cần kết hợp với confidence threshold để tránh ghi đè bằng fact nhiễu."
    )


if __name__ == "__main__":
    main()
