from dataclasses import dataclass
from time import time
from typing import Optional

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.state import ResearchState


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: Optional[str] = None


def should_stop(state: ResearchState, settings: Settings) -> StopDecision:
    if state.iteration >= settings.max_iterations:
        return StopDecision(stop=True, reason="max_iterations_exceeded")

    if state.deadline_at is not None and time() > state.deadline_at:
        return StopDecision(stop=True, reason="timeout_exceeded")

    return StopDecision(stop=False)
