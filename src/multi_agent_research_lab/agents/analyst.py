"""Analyst agent skeleton."""

from time import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.validation import validate_agent_output
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        state.add_trace_event("agent.start", {"agent": self.name, "iteration": state.iteration, "ts": time()})
        with trace_span("agent.analyst") as span:
            system_prompt = (
                "You are an analyst agent. Review research findings and provide key themes, insights, and recommendations."
            )
            user_prompt = f"""Research notes:
{state.research_notes}

Provide:
1. Key themes and patterns
2. Important insights
3. Gaps or contradictions
4. Recommendations for the writer"""

            response_content = ""
            input_tokens = None
            output_tokens = None
            cost_usd = None
            try:
                response = self.llm.complete(system_prompt, user_prompt)
                response_content = response.content
                input_tokens = response.input_tokens
                output_tokens = response.output_tokens
                cost_usd = response.cost_usd
            except Exception as exc:
                state.errors.append(f"analyst.llm_failed:{type(exc).__name__}")
                state.add_trace_event(
                    "agent.error",
                    {"agent": self.name, "stage": "llm", "error_type": type(exc).__name__, "ts": time()},
                )
                response_content = f"Unable to analyze due to LLM error: {type(exc).__name__}"

            if not response_content.strip():
                state.errors.append("analyst.empty_output")
                response_content = "No analysis generated."

            state.analysis_notes = response_content
            span["attributes"]["input_tokens"] = input_tokens
            span["attributes"]["output_tokens"] = output_tokens
            span["attributes"]["cost_usd"] = cost_usd

            validation_errors = validate_agent_output(self.name, state)
            if validation_errors:
                state.errors.extend(validation_errors)
                state.add_trace_event(
                    "agent.validation",
                    {"agent": self.name, "errors": validation_errors, "ts": time()},
                )

        state.add_trace_event("agent.end", {"agent": self.name, "duration_seconds": span.get("duration_seconds"), "ts": time()})

        state.agent_results.append(AgentResult(
            agent=AgentName.ANALYST,
            content=state.analysis_notes or "",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "duration_seconds": span.get("duration_seconds"),
            }
        ))

        return state
