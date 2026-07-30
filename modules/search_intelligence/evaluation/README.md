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

## Parameterized stress benchmark

The stress suite expands the smoke test to 24 matched scenarios: three map
layouts, near/far target placements, and correct/diffuse/uniform/misleading
priors. Five paired profiles vary sensor recall, false alarms, observation
quality, and the viewpoint budget.

```bash
python run/run_search_stress_benchmark.py \
  --repetitions 20 \
  --seed 20260730 \
  --output-dir results/search_stress_benchmark
```

This runs 9,600 episodes with the default four policies. It writes a combined
episode CSV, grouped summary CSV, JSON manifest, and a complete trace report for
each profile. Profile comparisons reuse the same scenario/repetition seeds so
sensor samples remain paired.

To isolate detection verification under nominal and high-false-alarm sensing:

```bash
python run/run_search_stress_benchmark.py \
  --profiles verified_nominal verified_high_false_alarm \
  --repetitions 20 \
  --seed 20260730 \
  --output-dir results/search_verification_benchmark_v2
```

The verified profiles require two observations of the same target entity before
declaring `FOUND`. Independent simulated false alarms receive viewpoint-specific
entity IDs and therefore cannot confirm one another.
