# GSI SearchWorld V2

SearchWorld V2 is a deterministic family of outdoor Gazebo/PX4 search maps.
It tests whether one task-conditioned belief policy transfers across physical
layouts without changing the robot, ROS topics, detector, or policy weights.

V2 uses a dedicated standard X500 airframe (`4011`) with one 640 x 480, 10 Hz
nadir RGB-D camera. The nadir mount matches the candidate generator's circular
ground-footprint model. V1.1 keeps its original 45-degree oblique camera.

The initial suite contains:

| Scenario | Character | Main difficulty | Fixed seed |
| --- | --- | --- | ---: |
| `campus` | Academic campus, parking, sports field, internal roads | Mixed open space and tall buildings | 101 |
| `industrial` | Warehouses, loading apron, container yard | Strong occlusion and repeated structures | 202 |
| `suburban` | Main street, houses, shops, park, school | Distributed semantic cues and vegetation | 303 |

Every configuration controls map size, seed, object counts, target slot,
flight/search parameters, and the task-conditioned semantic prior. Ground
truth is written separately and is never loaded by the search policy.

## Generate and inspect

```bash
cd /tmp/GSI/ros2_ws
bash generate_search_world_v2.sh all
```

Each scenario writes an SDF world, public semantic map, prior fixture, private
ground truth, ROS parameters, Gazebo bridge, VisionFlow profile, and a SHA-256
manifest below `simulation/search_world_v2/<scenario>/generated/`.

## Install and run one map

```bash
bash install_search_world_v2.sh /home/windylab/workspace/VisionFlow-PX4
bash start_search_world_v2_sitl.sh campus foreground
```

In another terminal inside the simulator container:

```bash
cd /tmp/GSI/ros2_ws
bash run_search_world_v2_search.sh campus
```

Replace `campus` with `industrial` or `suburban` for the other maps.

## Repeated three-map experiment

```bash
GSI_REPETITIONS=3 GSI_TRIAL_TIMEOUT_S=600 \
  bash run_search_world_v2_batch.sh
```

The runner cleanly restarts PX4/Gazebo for every trial. It saves the exact
scenario/configuration snapshot, manifest, ground truth, simulator/runtime
logs, complete viewpoint trace, per-trial summary, aggregate `trials.csv`, and
`batch_summary.json` under `results/gazebo_search_world_v2/<timestamp>/`.

V2 priors use `projection_mode: label_mass`: each semantic category's weight
is divided across the cells covered by that category before normalization.
This prevents a large parking polygon from receiving more total prior mass
only because it contains more grid cells. Legacy worlds retain the original
`cell_affinity` projection unless they opt in explicitly.

Batch experiments use `HEADLESS=1` by default so the Gazebo GUI cannot distort
the simulator real-time factor or resource measurements. Set `GSI_HEADLESS=0`
for a visible diagnostic batch. The single-map command remains visible by
default.

The three checked-in seeds are the V2 development set. New seeds and held-out
target slots should be added as a separate evaluation set rather than tuning
the policy against these three maps.
