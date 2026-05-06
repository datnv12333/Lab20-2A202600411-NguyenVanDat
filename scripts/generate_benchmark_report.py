from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.services.llm_client import LLMClient


def _parse_benchmark_results(text: str) -> dict:
    import re
    from typing import Optional

    query_blocks = []
    parts = text.split("\n--------------------------------------------------------------------------------\n")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        m = re.search(r"^## Query (\d+): (.+)$", part, flags=re.MULTILINE)
        if not m:
            continue
        query_index = int(m.group(1))
        query_text = m.group(2).strip()

        def _grab_float(label: str) -> Optional[float]:
            mm = re.search(rf"^- {re.escape(label)}: ([0-9.]+)s?$", part, flags=re.MULTILINE)
            if not mm:
                return None
            try:
                return float(mm.group(1))
            except Exception:
                return None

        baseline_latency = _grab_float("Baseline latency")
        baseline_quality = _grab_float("Baseline quality (heuristic)")
        baseline_cost = _grab_float("Baseline cost (estimated)")
        multi_latency = _grab_float("Multi-agent latency")
        multi_quality = _grab_float("Multi-agent quality (heuristic)")
        multi_cost = _grab_float("Multi-agent cost (estimated)")

        query_blocks.append({
            "index": query_index,
            "query": query_text,
            "baseline": {"latency_s": baseline_latency, "quality": baseline_quality, "cost_usd": baseline_cost},
            "multi_agent": {"latency_s": multi_latency, "quality": multi_quality, "cost_usd": multi_cost},
        })

    return {"queries": sorted(query_blocks, key=lambda x: x["index"])}


def _fetch_langsmith_trace_snapshot(project_name: str, limit: int = 20) -> dict:
    from datetime import datetime, timezone
    from typing import Optional

    try:
        from langsmith import Client
    except Exception:
        return {"enabled": False, "reason": "langsmith_not_installed"}

    client = Client()
    runs = list(client.list_runs(project_name=project_name, is_root=True, limit=limit))
    if not runs:
        return {"enabled": True, "project": project_name, "root_runs_found": 0}

    runs_sorted = sorted(
        runs,
        key=lambda r: (getattr(r, "start_time", None) or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )

    latest = None
    for r in runs_sorted:
        name = getattr(r, "name", "") or ""
        if name == "workflow.invoke" or "workflow.invoke" in name:
            latest = r
            break
    if latest is None:
        latest = runs_sorted[0]

    trace_id = getattr(latest, "trace_id", None)
    trace_runs = []
    if trace_id is not None:
        trace_runs = list(client.list_runs(project_name=project_name, trace_id=trace_id, limit=200))

    def _duration_seconds(run) -> Optional[float]:
        st = getattr(run, "start_time", None)
        en = getattr(run, "end_time", None)
        if st is None or en is None:
            return None
        try:
            return float((en - st).total_seconds())
        except Exception:
            return None

    def _run_dict(run) -> dict:
        return {
            "id": str(getattr(run, "id", "")),
            "name": getattr(run, "name", None),
            "run_type": getattr(run, "run_type", None),
            "error": bool(getattr(run, "error", False)),
            "start_time": str(getattr(run, "start_time", "")),
            "duration_seconds": _duration_seconds(run),
            "parent_run_id": str(getattr(run, "parent_run_id", "")) if getattr(run, "parent_run_id", None) else None,
        }

    step_runs = []
    for r in trace_runs:
        name = getattr(r, "name", "") or ""
        if name.startswith("agent.") or name in {"workflow.invoke"}:
            step_runs.append(_run_dict(r))

    return {
        "enabled": True,
        "project": project_name,
        "root_runs_found": len(runs),
        "latest_root_run": _run_dict(latest),
        "latest_trace_id": str(trace_id) if trace_id is not None else None,
        "latest_trace_run_count": len(trace_runs),
        "latest_trace_steps": sorted(
            step_runs,
            key=lambda x: (x.get("start_time") or ""),
        ),
    }


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    settings = get_settings()

    from pathlib import Path

    results_path = Path("reports/benchmark_results.txt")
    if not results_path.exists():
        raise SystemExit("Missing reports/benchmark_results.txt. Run scripts/run_benchmark.py first.")

    text = results_path.read_text(encoding="utf-8")
    parsed = _parse_benchmark_results(text)
    query_count = len(parsed.get("queries", []))

    trace_snapshot = _fetch_langsmith_trace_snapshot(settings.langsmith_project)

    summary_path = Path("reports/benchmark_summary.md")
    if not settings.openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Provide it in .env to generate LLM summary.")

    llm = LLMClient(provider="openai")
    prompt = f"""Bạn là một senior ML/LLM engineer viết benchmark report chuyên nghiệp cho multi-agent systems.

Mục tiêu:
- So sánh single-agent baseline vs multi-agent theo metrics một cách định lượng.
- Đưa nhận xét về trace mới nhất từ LangSmith (nếu có) để giải thích “ai làm gì, tốn bao nhiêu, sai ở đâu”.
- Kết luận và khuyến nghị rõ ràng, thực dụng.

Yêu cầu output (Markdown):
1) Executive Summary (3-6 bullets)
2) Metrics Comparison
   - Bảng theo từng query với các cột bắt buộc:
     - Baseline latency (s), Multi latency (s), ΔLatency = Multi - Baseline (s)
     - Baseline quality, Multi quality, ΔQuality = Multi - Baseline
     - Baseline cost (USD), Multi cost (USD), ΔCost = Multi - Baseline (USD)
   - Quy ước dấu:
     - ΔLatency > 0 nghĩa là multi-agent chậm hơn baseline
     - ΔQuality > 0 nghĩa là multi-agent tốt hơn
     - ΔCost > 0 nghĩa là multi-agent đắt hơn
   - Bảng tổng hợp: average + median cho từng metric (nếu đủ dữ liệu) và nhận xét ngắn gọn
   - Format số: latency 2 chữ số thập phân, cost 6 chữ số thập phân (nếu có), quality 1 chữ số thập phân
3) Trace Observations (LangSmith)
   - Số lượng query trong benchmark: {query_count}
   - Nếu có LangSmith trace:
     - nêu latest root run (id, duration)
     - nêu số lượng root runs gần nhất tìm thấy trong project (từ snapshot)
     - tổng số run trong trace
     - liệt kê các step chính (workflow.invoke, agent.*) và duration
     - top 3 agent step tốn thời gian nhất (chỉ xét agent.*) + % so với workflow.invoke (ước tính)
   - Nhận xét: bottleneck nằm ở step nào, có error không, routing có hợp lý không
4) Failure Modes & Reliability
5) Recommendations
   - Khi nào dùng baseline vs multi-agent
   - Những cải tiến tiếp theo ưu tiên (2-5 bullets)

Chỉ dùng dữ liệu dưới đây, không bịa số:

Benchmark raw text:
{text}

Benchmark parsed (JSON):
{parsed}

LangSmith trace snapshot (JSON, có thể thiếu nếu chưa bật tracing hoặc không có run):
{trace_snapshot}
"""
    response = llm.complete("You write benchmark summaries.", prompt)
    summary_path.write_text(response.content, encoding="utf-8")

    print(str(summary_path))


if __name__ == "__main__":
    main()
