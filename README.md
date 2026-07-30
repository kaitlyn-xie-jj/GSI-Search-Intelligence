# GSI Search Intelligence

GSI is a research framework for task-conditioned autonomous target search in
open outdoor environments. Given a natural-language task and a semantic map,
it builds a spatial prior, maintains a Bayesian target belief, selects the next
viewpoint, and sends the resulting goal to a robot navigation stack.

The current integration targets one PX4 UAV in Gazebo Harmonic through ROS 2
Humble and MAVROS.

## Architecture

```text
Natural-language task
        |
        v
Task-conditioned semantic prior
        |
        v
Bayesian belief + semantic search grid
        |
        v
ActiveSearchPolicy -> next viewpoint
        |
        v
MAVROS Offboard -> PX4 -> Gazebo sensors
        |
        +---- observation and belief update ----+
```

The upper layer decides what and where to search. PX4 and the platform planner
remain responsible for vehicle control and execution.

## Included Components

- Search contracts: `SearchTask`, `SearchObservation`, `SearchState`, and
  `SearchOutcome`.
- Coverage, random, greedy-prior, and belief-aware active-search policies.
- Bayesian positive and negative observation updates.
- Semantic-map annotation and task-conditioned prior projection.
- Policy benchmark and evaluation utilities.
- ROS 2 adapters for odometry, RGB, depth, point cloud, detections, and battery.
- MAVROS Offboard controller with staged takeoff and bounded horizontal
  setpoint progression.
- Parameterized GSI SearchWorld V1 for Gazebo/PX4 experiments.

## Repository Layout

```text
modules/search_intelligence/   Platform-neutral search contracts and policies
ros2_ws/src/gsi_search_bridge/ ROS 2, MAVROS, perception, and Gazebo adapters
ros2_ws/simulation/            SearchWorld configuration and generated artifacts
run/                           Offline experiment and benchmark entry points
docs/                          Supporting project documentation
```

VisionFlow-PX4 is an external simulator dependency and is intentionally kept in
a separate repository.

## Run SearchWorld V1

Prerequisites:

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Harmonic 8
- PX4 SITL and MAVROS from VisionFlow-PX4

Generate and install the world:

```bash
bash ros2_ws/generate_search_world_v1.sh
bash ros2_ws/install_search_world_v1.sh /workspace/VisionFlow-PX4
```

Start PX4 and Gazebo from the VisionFlow-PX4 checkout:

```bash
bash docker/run_gz_sitl.sh --profile "GSI SearchWorld V1"
```

Start the live search inside the compatible ROS/PX4 container:

```bash
bash /tmp/GSI/ros2_ws/run_search_world_v1_search.sh
```

The search result is published as JSON on `/gsi/search/outcome`.

## Tests

Run the platform-neutral tests from the repository root:

```bash
python -m unittest discover -s modules/search_intelligence/tests -p "test_*.py"
```

Collect the parameterized search-policy stress dataset:

```bash
python run/run_search_stress_benchmark.py \
  --repetitions 20 \
  --seed 20260730 \
  --output-dir results/search_stress_benchmark
```

The default matrix covers 24 map/target/prior scenarios and five sensor/resource
profiles, producing 9,600 paired policy episodes.

Build and test the ROS package in the ROS 2 environment:

```bash
cd ros2_ws
colcon build --symlink-install --packages-select gsi_search_bridge
colcon test --packages-select gsi_search_bridge
```

## Current Status

SearchWorld V1 has completed a live task-conditioned search using Gazebo sensor
data, Bayesian belief updates, active viewpoint selection, PX4 Offboard flight,
target detection, and RTL. The current semantic prior is a deterministic fixture
that follows the intended LLM output contract; live LLM prior generation and
cross-scenario calibration remain future evaluation work.

The bundled color detector is a simulator interface baseline, not the final
open-world perception method. The current VisionFlow airframe also includes a
manipulator and should be replaced by a landing-stable search UAV model for
repeatable touchdown experiments.

More details are available in
[`ros2_ws/src/gsi_search_bridge/README.md`](ros2_ws/src/gsi_search_bridge/README.md)
and
[`ros2_ws/simulation/search_world_v1/README.md`](ros2_ws/simulation/search_world_v1/README.md).
The 2026-07-30 clean-restart, PX4 preflight recovery, and live sensor-frame
evidence is recorded in
[`docs/zh/validation/gazebo-searchworld-v1-2026-07-30.md`](docs/zh/validation/gazebo-searchworld-v1-2026-07-30.md).

The complete mathematical specification, policy equations, benchmark design,
and statistical estimators are documented in
[`docs/zh/concepts/search-intelligence-mathematics.md`](docs/zh/concepts/search-intelligence-mathematics.md).
