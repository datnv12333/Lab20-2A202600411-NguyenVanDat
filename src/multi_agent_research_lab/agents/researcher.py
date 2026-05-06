"""Researcher agent skeleton."""

from time import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.validation import validate_agent_output
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm_client: LLMClient, search_client: SearchClient):
        self.llm = llm_client
        self.search = search_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources
        settings = get_settings()

        state.add_trace_event("agent.start", {"agent": self.name, "iteration": state.iteration, "ts": time()})
        with trace_span("agent.researcher") as span:
            try:
                sources = self.search.search(query, max_results=max_sources)
            except Exception as exc:
                sources = []
                state.errors.append(f"researcher.search_failed:{type(exc).__name__}")
                state.add_trace_event(
                    "agent.error",
                    {"agent": self.name, "stage": "search", "error_type": type(exc).__name__, "ts": time()},
                )

            state.sources = sources
            search_cost_usd = settings.search_cost_usd_per_call if sources else None
            if search_cost_usd is not None:
                span["attributes"]["search_cost_usd"] = search_cost_usd

            search_results = "\n\n".join([
                f"[{i+1}] {s.title}\nURL: {s.url}\n{s.snippet}"
                for i, s in enumerate(sources)
            ])

            system_prompt = (
                "You are a research agent. Extract key findings from search results and compile structured research notes."
            )
            user_prompt = f"""Query: {query}

Search results:
{search_results}

Provide 3-5 key findings with source citations in this format:
1. [Finding] - Source: [Title/URL]"""

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
                state.errors.append(f"researcher.llm_failed:{type(exc).__name__}")
                state.add_trace_event(
                    "agent.error",
                    {"agent": self.name, "stage": "llm", "error_type": type(exc).__name__, "ts": time()},
                )

                if sources:
                    response_content = "\n".join([
                        f"{i+1}. {s.title} - Source: {s.url or s.title}"
                        for i, s in enumerate(sources[:5])
                    ])
                else:
                    response_content = "No sources available to compile research notes."

            if not response_content.strip():
                state.errors.append("researcher.empty_output")
                response_content = "No research notes generated."

            state.research_notes = response_content
            span["attributes"]["sources_count"] = len(sources)
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
            agent=AgentName.RESEARCHER,
            content=state.research_notes or "",
            metadata={
                "sources_count": len(state.sources),
                "search_calls": 1,
                "search_cost_usd": search_cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "duration_seconds": span.get("duration_seconds"),
            }
        ))

        return state
