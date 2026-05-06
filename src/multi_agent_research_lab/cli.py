"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.metrics import summarize_metrics
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    from multi_agent_research_lab.services.llm_client import LLMClient

    llm = LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    prompt = f"""Research this query, analyze findings, and write a 500-word summary:
{query}

Include sources and structure your answer clearly."""

    response = llm.complete("You are a helpful research assistant.", prompt)
    state.final_answer = response.content

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    from multi_agent_research_lab.services.llm_client import LLMClient
    from multi_agent_research_lab.services.search_client import SearchClient

    llm = LLMClient()
    search = SearchClient()

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(llm, search)
    try:
        result = workflow.run(state)
        console.print(Panel.fit(result.final_answer or "No answer", title="Multi-Agent Result"))
        metrics = summarize_metrics(result)
        console.print(Panel.fit(metrics, title="Run Metrics"))
        from multi_agent_research_lab.observability.trace_export import write_trace

        path = write_trace("reports/last_trace.json", result)
        console.print(Panel.fit(str(path), title="Trace Saved"))
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
