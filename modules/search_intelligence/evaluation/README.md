# Search Policy Benchmark (M6)

This package evaluates search policies without importing the full semantic platform
or visualization stack. It reuses the production `SearchTask`, `SearchState`,
`SearchObservation`, `SearchOutcome`, `SearchSession`, and Bayesian belief updater.

## Compared policies

- `coverage`: legacy zigzag route with configurable observation sampling
- `random`: deterministic seeded candidate ordering
- `greedy_prior`: fixed initial-prior ranking
- `active`: posterior-aware detection, information, novelty, and travel utility

## Fairness controls

Every policy receives the same task, grid, initial belief, target placement, sensor
model, movement model, footprint, and resource budget. Sensor outcomes use a stable
hash of the suite seed, scenario, repetition, and viewpoint. Therefore, two policies
visiting the same viewpoint in the same repetition receive the same sensor sample.

Simulator ground truth is used only by the observation generator and metric scorer.
It is never included in `SearchTask`, the initial belief, or policy state.

## Metrics

- true target success rate and declared-found rate
- false-positive rate
- elapsed time, distance, energy, and viewpoint count
- coverage fraction
- SPL using the shortest candidate distance capable of observing the target
- initial/final belief entropy and entropy reduction
- normal-approximation 95% confidence intervals

Metrics are aggregated overall and by prior condition. The included smoke scenarios
cover correct, uniform, noisy, and misleading priors.

## Run

```bash
python run/run_search_policy_benchmark.py \
  --repetitions 20 \
  --seed 20260728 \
  --output-dir results/search_policy_benchmark
```

The command writes a complete JSON report, a per-episode CSV, and an aggregate CSV.
The bundled scenarios validate the benchmark pipeline; they are not a publication
dataset and their scores must not be presented as final research results.
