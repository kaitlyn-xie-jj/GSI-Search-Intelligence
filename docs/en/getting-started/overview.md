# Project Overview

GSI is designed for task planning, allocation, execution validation, and replanning in multi-robot environments. It connects natural-language tasks, scene state, robot capabilities, and execution feedback into benchmark and training workflows.

## Core Capabilities

- Convert task instructions and environment state into structured plans.
- Use validators to check plan format, skill constraints, and state constraints.
- Use the TANGO allocator to assign ready tasks to robots.
- Execute skills in the semantic platform and record results.
- Trigger replanning after execution failures or new cases.
- Collect SFT/RLVR data and evaluate trained models.

## Typical Flow

```text
Dataset task
  -> SGI planner
  -> validator / allocator
  -> semantic platform
  -> feedback / replan
  -> metrics / outputs
```

## Next Steps

- Run a minimal example: [Quick Start](quickstart.md).
- Reproduce experiments: [Reproduce Results](reproduce-results.md).
- Understand the codebase: [Repository Layout](../development/repo-layout.md).
