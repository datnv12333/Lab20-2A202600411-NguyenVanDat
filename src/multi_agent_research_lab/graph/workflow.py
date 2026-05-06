"""LangGraph workflow skeleton."""

from time import time

from langgraph.graph import StateGraph, END

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self, llm_client: LLMClient, search_client: SearchClient):
        self.supervisor = SupervisorAgent(llm_client=llm_client)
        self.researcher = ResearcherAgent(llm_client, search_client)
        self.analyst = AnalystAgent(llm_client)
        self.writer = WriterAgent(llm_client)
        self.critic = CriticAgent(llm_client)
        self.graph = None

    def build(self) -> object:
        """Create a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)
        workflow.add_node("critic", self.critic.run)

        workflow.set_entry_point("supervisor")

        def route_supervisor(state: ResearchState) -> str:
            return state.route_history[-1] if state.route_history else "END"

        workflow.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "END": END
            }
        )

        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", END)

        self.graph = workflow.compile()
        return self.graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        if not self.graph:
            self.build()

        state.add_trace_event("workflow.start", {"started_at": time()})
        with trace_span("workflow.invoke") as span:
            result = self.graph.invoke(state)
        state.add_trace_event("workflow.invoke", span)
        state.add_trace_event("workflow.end", {"ended_at": time()})

        if isinstance(result, dict):
            return ResearchState(**result)
        return result
