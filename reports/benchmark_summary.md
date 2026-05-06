# Benchmark Report: Single-Agent vs Multi-Agent Systems

## Executive Summary
- The benchmark compares single-agent and multi-agent systems across three queries, focusing on latency, quality, and cost.
- Multi-agent systems exhibit higher latency and cost but significantly improved quality in all queries.
- The latest LangSmith trace indicates a well-structured workflow with no errors, highlighting the efficiency of the multi-agent approach.
- Key bottlenecks were identified in specific agent tasks, particularly in the writing phase, which contributed to increased latency.
- Recommendations include strategic use of multi-agent systems for complex tasks while considering cost and latency trade-offs.

## Metrics Comparison

| Query | Baseline Latency (s) | Multi Latency (s) | ΔLatency (s) | Baseline Quality | Multi Quality | ΔQuality | Baseline Cost (USD) | Multi Cost (USD) | ΔCost (USD) |
|-------|-----------------------|-------------------|---------------|------------------|---------------|----------|---------------------|------------------|--------------|
| 1     | 13.98                 | 43.63             | 29.65         | 6.0              | 9.0           | 3.0      | 0.00048795          | 0.00231165       | 0.00182370   |
| 2     | 12.55                 | 40.42             | 27.87         | 6.0              | 9.0           | 3.0      | 0.00050400          | 0.00200685       | 0.00150285   |
| 3     | 14.84                 | 39.85             | 25.01         | 4.0              | 9.0           | 5.0      | 0.00058500          | 0.00219510       | 0.00161010   |

### Summary Statistics
- **Average Latency**: 
  - Baseline: 13.79s
  - Multi-Agent: 41.30s
  - ΔLatency: 27.51s
- **Average Quality**: 
  - Baseline: 5.33
  - Multi-Agent: 9.0
  - ΔQuality: 3.67
- **Average Cost**: 
  - Baseline: 0.00052532 USD
  - Multi-Agent: 0.00217187 USD
  - ΔCost: 0.00164655 USD

### Observations
- Multi-agent systems are slower and more expensive but provide significantly higher quality outputs, making them suitable for complex tasks.

## Trace Observations (LangSmith)
- **Total Queries in Benchmark**: 3
- **Latest Root Run**: 
  - ID: `ad7eda56-85a0-45d4-ace8-ee006b1bfebf`
  - Duration: 39.83s
- **Total Runs in Trace**: 26
- **Key Steps**:
  - `workflow.invoke`: 39.83s
  - `agent.researcher`: 7.56s
  - `agent.analyst`: 7.79s
  - `agent.writer`: 16.99s
  - `agent.critic`: 3.20s
- **Top 3 Time-Consuming Steps**:
  - `agent.writer`: 16.99s (42.6% of total duration)
  - `workflow.invoke`: 39.83s (100% of total duration)
  - `agent.analyst`: 7.79s (19.5% of total duration)

### Observations
- The bottleneck is primarily in the `agent.writer` step, which significantly contributes to the overall latency. No errors were detected, and routing appears efficient.

## Failure Modes & Reliability
- The multi-agent system demonstrated reliability with no errors reported during the execution of tasks. However, the increased latency indicates potential areas for optimization, particularly in the writing and analysis phases.

## Recommendations
- **When to Use Baseline vs Multi-Agent**:
  - Use single-agent systems for straightforward, low-complexity tasks where latency and cost are critical.
  - Opt for multi-agent systems for complex tasks requiring high-quality outputs and where the benefits of improved quality outweigh the costs and latency.
  
- **Prioritized Improvements**:
  - Optimize the `agent.writer` process to reduce latency.
  - Explore caching mechanisms for frequently requested information to improve response times.
  - Implement more efficient routing strategies to minimize handoff delays between agents.
  - Consider hybrid models that leverage both single-agent and multi-agent systems based on task complexity.

By following these recommendations, organizations can effectively balance the trade-offs between cost, latency, and quality in their AI deployments.

## Trace
![Trace](./reports/trace.png)
