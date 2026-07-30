# Evaluation Metrics

GSI benchmark records task success, latency, LLM calls, replanning, new cases, and resource usage. Field names may vary across experiments, but the core meanings remain consistent.

## Core Metrics

| Metric | Meaning |
| --- | --- |
| `success_rate` | Ratio of successful runs to total runs. |
| `elapsed_sec` | Total runtime of a single task. |
| `planning_duration` | Time spent generating or updating a plan. |
| `allocation_duration` | Time spent by the TANGO allocator. |
| `llm_calls` | Number of LLM calls in one task run. |
| `prompt_tokens_total` | Total prompt tokens; requires token statistics. |
| `response_tokens_total` | Total response tokens; requires token statistics. |
| `total_energy` | Estimated total execution energy or movement cost. |
| `replans_total` | Full and partial replans combined. |
| `newcase_total` | Number of new cases recorded during execution. |
| `atomic_task_completion_rate` | Completed atomic tasks divided by total atomic tasks. |

## Usage

Do not compare methods using only `success_rate`. Also compare latency, LLM calls, replans, and energy. For new-case evaluation, compare `newcase_total`, `newcase_top_types`, and changes across `newcase_N` directories.

## Output Location

Aggregate metrics are usually located at:

```text
<output-root>/<timestamp>/batch_<method>/aggregate_full.json
<output-root>/<timestamp>/newcase_N/batch_<method>/aggregate_full.json
```

See [Output Directory](../training/outputs.md) for structure details.
