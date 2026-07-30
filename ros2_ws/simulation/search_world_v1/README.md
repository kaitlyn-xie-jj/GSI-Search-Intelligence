# GSI SearchWorld V1

SearchWorld V1 is a deterministic outdoor benchmark shared by Gazebo physics,
GSI semantic search, and the evaluator. One JSON config generates all artifacts;
the policy never reads the evaluator-only ground truth.

## Generated artifacts

```text
search_world_v1.json
  -> generated/gsi_search_world_v1.sdf   Gazebo physics and visuals
  -> generated/semantic_map.json         public GSI scene graph
  -> generated/search_prior.json         task-conditioned prior fixture
  -> generated/ground_truth.json         evaluator only
  -> generated/search_params.yaml        ROS search and detector parameters
  -> generated/gz_bridge.yaml            world-specific OakD topic bridge
  -> generated/visionflow_profile.conf   VisionFlow profile fragment
  -> generated/scenario_manifest.json    provenance and SHA-256 hashes
```

The default 80 m by 60 m ENU world contains parking, loading, park, campus,
entrance, road, and restricted semantic regions. Buildings, trees, parked
vehicles, containers, fences, and the target provide physical geometry and
occlusion. The UAV flies at 12 m, above all V1 obstacles; obstacle-aware routing
is intentionally deferred to a later milestone.

## Parameterization

Edit `search_world_v1.json` to change:

- `world.seed` for deterministic procedural placement;
- `world.size_m` for the search bounds;
- `search.*` for grid, altitude, footprint, and viewpoint budget;
- `complexity.*` for trees, parked vehicles, and containers;
- `target.slot_index` for one of four controlled placements (`-1` selects by seed);
- `semantic_prior.*` for the upper-layer task-conditioned prior fixture.

Generate the artifacts:

```bash
bash ros2_ws/generate_search_world_v1.sh
```

Install the generated world and idempotent profile into the mounted VisionFlow
checkout from the SITL container:

```bash
bash /tmp/GSI/ros2_ws/install_search_world_v1.sh /workspace/VisionFlow-PX4
```

Start PX4 and Gazebo from WSL:

```bash
cd ~/workspace/VisionFlow-PX4
bash docker/run_gz_sitl.sh --profile "GSI SearchWorld V1"
```

Then start the live search from the container:

```bash
bash /tmp/GSI/ros2_ws/run_search_world_v1_search.sh
```

`semantic_map.json` intentionally contains no prop or target pose. Only an
offline evaluator may open `ground_truth.json`; policy code receives semantic
regions and the task-conditioned prior through `scenario_context.py`.
