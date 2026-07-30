# GSI Documentation

GSI is a benchmark and training workflow for multi-agent task planning, execution validation, solver-backed allocation, and model-based replanning.

## Documentation

- [English documentation](en/index.md)
- [中文文档](zh/index.md)

The English and Chinese documentation use the same structure. Chinese content remains available under `docs/zh/`; the English version is maintained under `docs/en/`.

## Local Development

Serve the documentation locally:

```bash
python -m mkdocs serve -a 127.0.0.1:8002
```

Build the static site:

```bash
python -m mkdocs build --strict
```

## Repository Entry Points

- `run/`: benchmark, evaluation, and data collection runners
- `modules/`: planner, validator, solver, world model, and platform logic
- `llm_finetune/`: SFT, RLVR, vLLM evaluation, and model workflow
- `docs/en/`: English documentation
- `docs/zh/`: Chinese documentation
