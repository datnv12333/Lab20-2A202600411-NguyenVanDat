import re
from typing import Optional, Tuple

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


_CITATION_RE = re.compile(r"(https?://\S+|\[\d+\])")


def total_tokens(state: ResearchState) -> Tuple[Optional[int], Optional[int]]:
    total_in = 0
    total_out = 0
    has_any = False
    for r in state.agent_results:
        in_t = r.metadata.get("input_tokens")
        out_t = r.metadata.get("output_tokens")
        if isinstance(in_t, int):
            total_in += in_t
            has_any = True
        if isinstance(out_t, int):
            total_out += out_t
            has_any = True
    if not has_any:
        return None, None
    return total_in, total_out


def total_cost_usd(state: ResearchState) -> Optional[float]:
    total = 0.0
    has_any = False
    for r in state.agent_results:
        c = r.metadata.get("cost_usd")
        if isinstance(c, (int, float)):
            total += float(c)
            has_any = True
        sc = r.metadata.get("search_cost_usd")
        if isinstance(sc, (int, float)):
            total += float(sc)
            has_any = True
    return total if has_any else None


def heuristic_quality_score(state: ResearchState) -> float:
    text = state.final_answer or ""
    words = len(text.split())
    score = 0.0

    if 350 <= words <= 650:
        score += 4.0
    elif 200 <= words < 350 or 650 < words <= 900:
        score += 2.0

    citations = len(_CITATION_RE.findall(text))
    if citations >= 3:
        score += 3.0
    elif citations >= 1:
        score += 1.5

    paragraphs = [p for p in text.split("\n") if p.strip()]
    if len(paragraphs) >= 4:
        score += 2.0
    elif len(paragraphs) >= 2:
        score += 1.0

    if state.errors:
        score -= min(3.0, 0.5 * len(state.errors))

    return max(0.0, min(10.0, score))


def build_metrics(run_name: str, latency_seconds: float, state: ResearchState) -> BenchmarkMetrics:
    in_t, out_t = total_tokens(state)
    cost = total_cost_usd(state)
    quality = heuristic_quality_score(state)
    notes = ""
    if in_t is not None and out_t is not None:
        notes = f"tokens_in={in_t}, tokens_out={out_t}, errors={len(state.errors)}"
    else:
        notes = f"errors={len(state.errors)}"
    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency_seconds,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=notes,
    )


def summarize_metrics(state: ResearchState) -> str:
    in_t, out_t = total_tokens(state)
    cost = total_cost_usd(state)
    quality = heuristic_quality_score(state)
    parts: list[str] = []
    if in_t is not None and out_t is not None:
        parts.append(f"Tokens: in={in_t}, out={out_t}")
    if cost is not None:
        parts.append(f"Estimated cost: ${cost:.4f}")
    parts.append(f"Heuristic quality: {quality:.1f}/10")
    parts.append(f"Errors: {len(state.errors)}")
    parts.append(f"Routes: {' -> '.join(state.route_history) if state.route_history else '(none)'}")
    return "\n".join(parts)
