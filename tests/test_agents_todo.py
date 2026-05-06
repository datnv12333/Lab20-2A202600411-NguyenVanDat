from time import time

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_in_order() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    agent = SupervisorAgent()

    state = agent.run(state)
    assert state.route_history[-1] == "researcher"

    state.research_notes = "notes"
    state = agent.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "analysis"
    state = agent.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "answer"
    state = agent.run(state)
    assert state.route_history[-1] == "critic"

    state.critic_notes = "critique"
    state = agent.run(state)
    assert state.route_history[-1] == "END"


def test_supervisor_stops_on_max_iterations(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ITERATIONS", "1")
    monkeypatch.setenv("TIMEOUT_SECONDS", "60")
    get_settings.cache_clear()

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 1
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "END"
    assert state.stopped_reason == "max_iterations_exceeded"


def test_supervisor_stops_on_timeout(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ITERATIONS", "10")
    monkeypatch.setenv("TIMEOUT_SECONDS", "60")
    get_settings.cache_clear()

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.started_at = time() - 100
    state.deadline_at = time() - 1
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "END"
    assert state.stopped_reason == "timeout_exceeded"
