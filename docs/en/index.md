# GSI Documentation

GSI is a system and training repository for embodied multi-agent task planning, execution validation, and replanning. The documentation covers evaluation, reproduction, model training, system extension, and troubleshooting.

## System Flow

```text
Task data
  -> LLM planner generates a task plan
  -> validator checks plan constraints
  -> TANGO allocator assigns robots
  -> semantic platform executes skills
  -> world model updates state
  -> feedback processor decides whether to replan
  -> batch runner aggregates metrics
```

## Entry Points

- First run: [Quick Start](getting-started/quickstart.md).
- Reproduction: [Reproduce Results](getting-started/reproduce-results.md).
- System design: [Architecture](concepts/architecture.md).
- Model training: [Training Overview](training/overview.md).
- Code extension: [Repository Layout](development/repo-layout.md).
- Troubleshooting: [GPU and vLLM](troubleshooting/gpu-and-vllm.md).

## Key Boundaries

- `solver_type` or `--methods` selects the planner method, such as `sgi`, `spine`, `smartllm`, or `lipllm`.
- `GSI_TANGO_SOLVER_BACKEND` selects the optimization backend for the TANGO allocator, such as `scip` or `gurobi`.
- vLLM provides an OpenAI-compatible model service; `GSI_LLM_MODEL` must match the served model name.
- `WindyLab/GSI` is the benchmark dataset. SFT and RLVR training use separate datasets.
- Standard benchmarks do not inject new cases by default. New-case evaluation requires `--enable-newcase` and `--newcase-counts`.

## Recommended Path

Run a small benchmark first, then review the output directory and architecture. Training depends on data, model checkpoints, validator, vLLM, and GPU configuration, so it should follow a stable evaluation setup.
