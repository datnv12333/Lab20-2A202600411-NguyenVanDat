"""Run benchmark comparing single-agent vs multi-agent workflows."""

from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def baseline_runner(query: str) -> ResearchState:
    llm = LLMClient()
    state = ResearchState(request=ResearchQuery(query=query))

    prompt = f"""Research this query, analyze findings, and write a 500-word summary:
{query}

Include sources and structure your answer clearly."""

    response = llm.complete("You are a helpful research assistant.", prompt)
    state.final_answer = response.content
    state.agent_results.append(AgentResult(
        agent=AgentName.WRITER,
        content=response.content,
        metadata={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    ))

    return state


def multi_agent_runner(query: str) -> ResearchState:
    llm = LLMClient()
    search = SearchClient()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow(llm, search)
    return workflow.run(state)


def main():
    load_dotenv(dotenv_path=".env", override=False)
    settings = get_settings()
    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents"
    ]

    results = []

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        print("\n[1/2] Running baseline...")
        baseline_state, baseline_metrics = run_benchmark("baseline", query, baseline_runner)
        print(f"Baseline completed in {baseline_metrics.latency_seconds:.2f}s")

        print("\n[2/2] Running multi-agent...")
        multi_state, multi_metrics = run_benchmark("multi-agent", query, multi_agent_runner)
        print(f"Multi-agent completed in {multi_metrics.latency_seconds:.2f}s")

        results.append({
            'query': query,
            'baseline_metrics': baseline_metrics.model_dump(),
            'multi_metrics': multi_metrics.model_dump(),
            'baseline_answer': baseline_state.final_answer,
            'multi_answer': multi_state.final_answer
        })

    # Generate report
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query'][:60]}...")
        b = result["baseline_metrics"]
        m = result["multi_metrics"]
        print(f"  Baseline:    {b['latency_seconds']:.2f}s, quality={b.get('quality_score')}, cost={b.get('estimated_cost_usd')}")
        print(f"  Multi-agent: {m['latency_seconds']:.2f}s, quality={m.get('quality_score')}, cost={m.get('estimated_cost_usd')}")
        print(f"  Difference:  {m['latency_seconds'] - b['latency_seconds']:.2f}s")

    # Save detailed report
    with open('reports/benchmark_results.txt', 'w') as f:
        f.write("# BENCHMARK REPORT: Single-Agent vs Multi-Agent\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"\n## Query {i}: {result['query']}\n\n")
            f.write(f"### Metrics\n")
            b = result["baseline_metrics"]
            m = result["multi_metrics"]
            f.write(f"- Baseline latency: {b['latency_seconds']:.2f}s\n")
            f.write(f"- Baseline quality (heuristic): {b.get('quality_score')}\n")
            f.write(f"- Baseline cost (estimated): {b.get('estimated_cost_usd')}\n")
            f.write(f"- Baseline notes: {b.get('notes')}\n")
            f.write(f"- Multi-agent latency: {m['latency_seconds']:.2f}s\n")
            f.write(f"- Multi-agent quality (heuristic): {m.get('quality_score')}\n")
            f.write(f"- Multi-agent cost (estimated): {m.get('estimated_cost_usd')}\n")
            f.write(f"- Multi-agent notes: {m.get('notes')}\n\n")
            f.write(f"### Baseline Answer\n{result['baseline_answer']}\n\n")
            f.write(f"### Multi-Agent Answer\n{result['multi_answer']}\n\n")
            f.write("-" * 80 + "\n")

    print("\nDetailed report saved to reports/benchmark_results.txt")


if __name__ == "__main__":
    main()
