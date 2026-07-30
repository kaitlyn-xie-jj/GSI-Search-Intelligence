# Development Guidelines

This page records current documentation and code maintenance conventions. New features should preserve module boundaries and runtime entry points.

## Module Boundaries

- Planner generates plans and does not handle low-level platform execution details.
- Validator checks plans and state and does not change training hyperparameters.
- Solver handles task allocation and optimization and does not own LLM prompt logic.
- Platform executes skills and returns feedback and does not decide global planning strategy.
- Batch runner executes batches and aggregates metrics and should not implicitly change task semantics.

## Configuration Priority

CLI arguments usually take precedence over `config/default.json`. Explicit training environment variables take precedence over defaults. Model service, dataset path, and solver backend should be declared in the run command.

## Documentation Updates

Update documentation when changing:

- Public script arguments.
- Data formats.
- Training defaults.
- Model or dataset repo ids.
- Validator / reward semantics.
- Output directory structure and metric fields.

## Commit Guidance

Keep large changes reviewable. Separate documentation structure, training scripts, business logic, and upstream code synchronization into distinct changes.
