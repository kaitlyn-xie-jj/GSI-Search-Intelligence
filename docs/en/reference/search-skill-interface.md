# Search Skill Interface

The Search Skill is a closed-loop, target-conditioned UAV capability. It selects
one short-horizon action at a time, consumes only observation-derived evidence,
and terminates when its success criteria or resource constraints are met.

## Input

| Required concept | Repository contract | Required content |
| --- | --- | --- |
| `EnvironmentModel` | `SearchGrid` | Searchable cells, public semantic labels, geometry, and excluded cells. It must not contain target ground truth. |
| `BeliefState` | `BeliefMap` / `SearchState.belief` | A normalized target-location probability over searchable cells. |
| `UAVState` | `Viewpoint` plus execution telemetry | Current map-frame pose, elapsed time, distance, energy, and trajectory-validity state. |
| `SensorModel` | `BinarySensorModel` and observation-quality fields | Conditional detector probability, false-positive probability, predicted visibility, sensor quality, and negative-update confidence. |
| `TargetSpec` | `SearchTask.target` and `SearchSuccessCriteria` | Open-vocabulary target query, attributes, confidence threshold, independent confirmations, persistence, and localization tolerance. |
| `Constraints` | `SearchBudget`, excluded regions, and candidate risk | Time, distance, energy, viewpoint limits, safety exclusions, speed assumptions, and completion reserve. |

`P(target in visible cells)`, `P(target visible | viewpoint)`, and
`P(sensor detects | target visible)` are separate values. The planner uses

```text
P(found | viewpoint)
  = P(target in visible cells)
  * P(target visible | viewpoint)
  * P(sensor detects | target visible)
```

The observation adapter must set `negative_update_strength` to zero and provide
`negative_update_rejection_reason` when RGB-D evidence is incomplete, point
projection is invalid, the view is blocked, or observation quality is below the
configured threshold.

## Output

Each planning cycle provides:

| Output | Contract |
| --- | --- |
| Next viewpoint | `SearchSession.next_viewpoint()` returning a map-frame `Viewpoint` |
| Short-horizon trajectory | `SearchSession.remaining_plan()`; the controller owns collision-free interpolation and execution |
| Updated belief | `SearchState.belief` after `record_observation` or `record_transit_observation` |
| Decision | `continue`, `confirm`, `success`, or `abort` as defined below |
| Explanation | `policy_decisions[-1]`, including probability terms, reward contributions, candidate-pool source, and final utility |

Decision semantics:

- `continue`: execute or retain the current trajectory; hysteresis rejected a replan.
- `confirm`: a positive detection exists but independent confirmation criteria are not yet met.
- `success`: `SearchOutcomeStatus.FOUND`; all configured confirmation and localization criteria are met.
- `abort`: budget exhausted, no candidates remain, the controller reports an unrecoverable invalid trajectory, or an external abort occurs.

## Assumptions and ownership

- Coordinates are map-frame metric coordinates. Sensor adapters own frame transforms.
- The UAV controller, not the search policy, guarantees collision avoidance and low-level flight stability.
- Visibility predictions may use public semantics, but never target pose or evaluator-only data.
- A circular footprint is a planning approximation only. It is not sufficient evidence for a Bayesian negative update.
- Positive evidence can be accepted in transit. Repeated adjacent frames do not count as independent confirmations.
- Replanning is mandatory for a positive detection or invalid trajectory. Negative evidence replans only after the configured interval and when belief, KL divergence, or expected reward changes enough.
- Terminal results include the final belief and resource metrics even when the target is not found.

