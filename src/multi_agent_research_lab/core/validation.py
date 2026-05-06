from __future__ import annotations

import re

from multi_agent_research_lab.core.state import ResearchState


_CITATION_RE = re.compile(r"(https?://\S+|\[\d+\])")


def validate_agent_output(agent: str, state: ResearchState) -> list[str]:
    errors: list[str] = []

    if agent == "researcher":
        if not state.sources:
            errors.append("researcher.no_sources")
        if not state.research_notes or not state.research_notes.strip():
            errors.append("researcher.missing_notes")
        elif not _CITATION_RE.search(state.research_notes):
            errors.append("researcher.missing_citations")

    if agent == "analyst":
        if not state.analysis_notes or len(state.analysis_notes.strip()) < 50:
            errors.append("analyst.too_short")

    if agent == "writer":
        if not state.final_answer or len(state.final_answer.strip()) < 200:
            errors.append("writer.too_short")
        elif not _CITATION_RE.search(state.final_answer):
            errors.append("writer.missing_citations")

    if agent == "critic":
        if not state.critic_notes or len(state.critic_notes.strip()) < 50:
            errors.append("critic.too_short")

    return errors
