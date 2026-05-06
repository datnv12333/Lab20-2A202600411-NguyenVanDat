"""Writer agent skeleton."""

from time import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.validation import validate_agent_output
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        state.add_trace_event("agent.start", {"agent": self.name, "iteration": state.iteration, "ts": time()})
        with trace_span("agent.writer") as span:
            system_prompt = "You are a writer agent. Create a comprehensive, well-structured answer with citations."
            user_prompt = f"""Original query: {state.request.query}

Research notes:
{state.research_notes}

Analysis notes:
{state.analysis_notes}

Write a 500-word answer with:
- Clear structure (intro, body, conclusion)
- Source citations
- Professional tone"""

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
                state.errors.append(f"writer.llm_failed:{type(exc).__name__}")
                state.add_trace_event(
                    "agent.error",
                    {"agent": self.name, "stage": "llm", "error_type": type(exc).__name__, "ts": time()},
                )
                response_content = f"Unable to write final answer due to LLM error: {type(exc).__name__}"

            if not response_content.strip():
                state.errors.append("writer.empty_output")
                response_content = "No final answer generated."

            state.final_answer = response_content
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
            agent=AgentName.WRITER,
            content=state.final_answer or "",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "duration_seconds": span.get("duration_seconds"),
            }
        ))

        return state
