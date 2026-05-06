"""Supervisor / router skeleton."""

import json
import re
from time import time
from typing import Optional

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.guardrails import should_stop
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def _route_deterministic(self, state: ResearchState) -> str:
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        if not state.critic_notes:
            return "critic"
        return "END"

    def _route_llm(self, state: ResearchState) -> tuple[str, str]:
        if self.llm is None:
            return self._route_deterministic(state), "llm_unavailable"

        allowed = ["researcher", "analyst", "writer", "critic", "END"]
        system_prompt = (
            "You are a supervisor/router for a multi-agent workflow. Choose the next step. "
            "Return ONLY JSON with key 'next' and value one of: researcher, analyst, writer, critic, END."
        )
        user_prompt = json.dumps({
            "allowed": allowed,
            "state": {
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
                "has_critic_notes": bool(state.critic_notes),
                "errors_count": len(state.errors),
                "last_route": state.route_history[-1] if state.route_history else None,
                "iteration": state.iteration,
            },
        })

        response = self.llm.complete(system_prompt, user_prompt)
        raw = response.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return self._route_deterministic(state), raw
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return self._route_deterministic(state), raw

        nxt = parsed.get("next")
        if nxt not in allowed:
            return self._route_deterministic(state), raw
        return str(nxt), raw

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()

        if state.started_at is None:
            state.started_at = time()
        if state.deadline_at is None:
            state.deadline_at = state.started_at + settings.timeout_seconds

        decision = should_stop(state, settings)
        if decision.stop:
            state.stopped_reason = decision.reason
            state.errors.append(decision.reason or "stopped")
            state.add_trace_event(
                "guard.stop",
                {"reason": decision.reason, "iteration": state.iteration, "max_iterations": settings.max_iterations},
            )
            state.record_route("END")
            return state

        if settings.supervisor_use_llm:
            next_route, raw = self._route_llm(state)
            state.add_trace_event(
                "supervisor.route_llm",
                {"next": next_route, "raw": raw[:5000], "iteration": state.iteration},
            )
        else:
            next_route = self._route_deterministic(state)

        state.add_trace_event("supervisor.route", {"next": next_route, "iteration": state.iteration})
        state.record_route(next_route)
        return state
