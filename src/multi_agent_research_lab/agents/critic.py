"""Optional critic agent skeleton for bonus work."""

from time import time

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.validation import validate_agent_output
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.
        """
        state.add_trace_event("agent.start", {"agent": self.name, "iteration": state.iteration, "ts": time()})
        with trace_span("agent.critic") as span:
            if not state.final_answer:
                state.critic_notes = "No final answer to critique."
                validation_errors = validate_agent_output(self.name, state)
                if validation_errors:
                    state.errors.extend(validation_errors)
                state.add_trace_event(
                    "agent.validation",
                    {"agent": self.name, "errors": validation_errors, "ts": time()},
                )
                state.add_trace_event(
                    "agent.end",
                    {"agent": self.name, "duration_seconds": span.get("duration_seconds"), "ts": time()},
                )
                state.agent_results.append(AgentResult(
                    agent=AgentName.CRITIC,
                    content=state.critic_notes or "",
                    metadata={"duration_seconds": span.get("duration_seconds")},
                ))
                return state

            sources_text = "\n\n".join([
                f"[{i+1}] {s.title}\nURL: {s.url}\n{s.snippet}"
                for i, s in enumerate(state.sources[:10])
            ])

            system_prompt = (
                "You are a critic agent. Your job is to review the final answer for factuality, "
                "citation coverage, and clarity. If there are issues, propose concrete fixes. "
                "If citations are missing or weak, request better citations using the provided sources."
            )
            user_prompt = f"""Original query: {state.request.query}

Available sources:
{sources_text}

Final answer:
{state.final_answer}

Return:
1) A short list of issues (bullets)
2) A short list of recommended fixes (bullets)
3) A revised final answer (only if needed). If not needed, return 'NO_CHANGES' for the revised answer."""

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
                state.errors.append(f"critic.llm_failed:{type(exc).__name__}")
                state.add_trace_event(
                    "agent.error",
                    {"agent": self.name, "stage": "llm", "error_type": type(exc).__name__, "ts": time()},
                )
                response_content = f"Critic failed due to LLM error: {type(exc).__name__}"

            state.critic_notes = response_content
            span["attributes"]["input_tokens"] = input_tokens
            span["attributes"]["output_tokens"] = output_tokens
            span["attributes"]["cost_usd"] = cost_usd

            if "NO_CHANGES" not in response_content:
                marker = "Revised final answer"
                idx = response_content.lower().find(marker.lower())
                if idx != -1:
                    revised = response_content[idx + len(marker):].strip(" :\n")
                    if revised:
                        state.final_answer = revised

            validation_errors = validate_agent_output(self.name, state)
            if validation_errors:
                state.errors.extend(validation_errors)
                state.add_trace_event(
                    "agent.validation",
                    {"agent": self.name, "errors": validation_errors, "ts": time()},
                )

        state.add_trace_event("agent.end", {"agent": self.name, "duration_seconds": span.get("duration_seconds"), "ts": time()})
        state.agent_results.append(AgentResult(
            agent=AgentName.CRITIC,
            content=state.critic_notes or "",
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "duration_seconds": span.get("duration_seconds"),
            }
        ))
        return state
