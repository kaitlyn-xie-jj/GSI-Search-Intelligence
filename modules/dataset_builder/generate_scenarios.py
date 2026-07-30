# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ==== project imports ====
try:
    from modules.config import unified_template_manager
    from modules.dataset_builder.scene_utils.scene_graph_builder import SceneGraphBuilder
    from modules.dataset_builder.scene_utils.generation_utils import (
        set_global_seed,
        rand_areas_per_district,
        rand_building_plan,
        parse_robot_plan_arg,
        rand_prop_plan,
        ensure_dir,
    )
    from modules.dataset_builder.scene_utils import (
        apply_perlin_to_areas_consistent,
        draw_scene,
    )
except Exception as e:
    raise ImportError(
        "Import failed. Please edit the import paths at the top of this file "
        "to match your project structure.\nOriginal error:\n" + str(e)
    )

DEFAULT_AREAS_PLAN = {
    "water_body": 2,
    "garden": 2,
    "neighborhood": 2,
    "square": 1,
    "industrial_park": 2,
    "campus": 2,
}

DEFAULT_BUILDING_PLAN = {
    "mall": 2,
    "hospital": 1,
    "library": 6,
    "power_station": 1,
    "hotel": 1,
    "robot_base": 1,
    "parking": 3,
}

DEFAULT_ROBOT_PLAN = {
    "UAV": 6,
    "FW_UAV": 6,
    "UGV": 6,
    "Quadruped": 6,
    "Humanoid": 6,
}

DEFAULT_PROP_PLAN = {
    "person": 20,
    "vehicle": 20,
    "cargo": 12,
    "boat": 6,
    "fire": 4,
    "equipment_failure": 2,
    "hazmat": 4,
    "assembly_component": 13,
}


# -----------------------------
# District layout helpers
# -----------------------------
def build_nine_district_layout(cell=300.0, origin=(0.0, 0.0)):
    ox, oy = origin

    def bounds(ix, iy):
        return {
            "x_min": ox + ix * cell,
            "y_min": oy + iy * cell,
            "x_max": ox + (ix + 1) * cell,
            "y_max": oy + (iy + 1) * cell,
        }

    layout = {
        "center_district": {"bounds": bounds(1, 1), "description": "Central district"},
        "north_district": {"bounds": bounds(1, 2), "description": "Northern district"},
        "south_district": {"bounds": bounds(1, 0), "description": "Southern district"},
        "west_district": {"bounds": bounds(0, 1), "description": "Western district"},
        "east_district": {"bounds": bounds(2, 1), "description": "Eastern district"},
        "north_west_district": {"bounds": bounds(0, 2), "description": "Northwestern district"},
        "north_east_district": {"bounds": bounds(2, 2), "description": "Northeastern district"},
        "south_west_district": {"bounds": bounds(0, 0), "description": "Southwestern district"},
        "south_east_district": {"bounds": bounds(2, 0), "description": "Southeastern district"},
    }
    return layout


def build_single_city_layout(world_size=900.0, origin=(0.0, 0.0)):
    ox, oy = origin
    return {
        "cybertown": {
            "bounds": {
                "x_min": ox,
                "y_min": oy,
                "x_max": ox + world_size,
                "y_max": oy + world_size,
            },
            "description": "Global city boundary (single layout)",
        }
    }


# --------------------------------
# Summary
# --------------------------------
def print_summary(nodes, edges):
    by_cat = defaultdict(int)
    by_type = defaultdict(int)
    for n in nodes:
        p = n.get("properties", {})
        by_cat[p.get("category", "NA")] += 1
        by_type[p.get("type", "NA")] += 1
    print("\n=== Scene Summary ===")
    print("Nodes by category:")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:15s}: {v}")
    print("Top node types:")
    for k, v in sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        print(f"  {k:15s}: {v}")
    print(f"Edges total: {len(edges)}")


# --------------------------------
# Utilities
# --------------------------------
def load_json_or_path(maybe_path: Optional[str]):
    """
    Generic parser supporting:
      - JSON string: '{"a":1}'
      - @path/to/file.json
      - None -> returns None
    """
    if not maybe_path:
        return None
    s = maybe_path.strip()
    if s.startswith("@"):
        path = s[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)


def resolve_areas_plan(
    user_areas_plan, district_layout: Dict[str, Dict], rng: random.Random
) -> Dict[str, Dict[str, int]]:
    """
    Parse area plan:
      - If user provides a complete per-district mapping object, use it directly;
      - If user provides a single-district generic object (type->count only), copy it to each district;
      - If user does not provide one, use the random generator.
    """
    if user_areas_plan is None:
        return rand_areas_per_district(district_layout, rng)

    # Check if every key is a district_id
    district_ids = set(district_layout.keys())
    keys = set(user_areas_plan.keys())

    if keys <= district_ids:
        # Full per-district plan
        return user_areas_plan

    # Otherwise treat as "single district generic plan"
    per_district = {}
    for d_id in district_layout.keys():
        per_district[d_id] = dict(user_areas_plan)  # Shallow copy is fine (values are counts)
    return per_district


def resolve_building_plan(
    user_building_plan, tmpl_mgr, rng: random.Random
) -> Dict[str, int]:
    """
    Building plan:
      - User provided: use directly (no forced randomization or constraints)
      - Not provided: random plan (via rand_building_plan)
    """
    if user_building_plan is not None:
        return user_building_plan
    return rand_building_plan(tmpl_mgr, rng)


def resolve_robot_plan(user_robot_plan_str: Optional[str]):
    """
    Robot plan: parse command-line string and return dict or None (falls back to SceneGraphBuilder default).
    Supports @path or JSON string.
    """
    return parse_robot_plan_arg(user_robot_plan_str)


def resolve_prop_plan(user_prop_plan, tmpl_mgr, rng: random.Random) -> Dict[str, int]:
    """
    Prop plan:
      - User provided: use directly
      - Not provided: random plan (via rand_prop_plan)
    """
    if user_prop_plan is not None:
        return user_prop_plan
    return rand_prop_plan(tmpl_mgr, rng)


# --------------------------------
# Main (single run)
# --------------------------------
def main(args):
    set_global_seed(args.seed)
    rng = random.Random(args.seed)

    # 1) layout
    if args.use_districts:
        district_layout = build_nine_district_layout(cell=args.cell, origin=(0.0, 0.0))
    else:
        district_layout = build_single_city_layout(
            world_size=args.world_size, origin=(0.0, 0.0)
        )

    # 2) plans: user-provided first, otherwise random
    if args.use_default_plans:
        # Area default: copy to each district
        areas_plan = {d: dict(DEFAULT_AREAS_PLAN) for d in district_layout.keys()}
        building_plan = dict(DEFAULT_BUILDING_PLAN)
        robot_plan = dict(DEFAULT_ROBOT_PLAN)
        prop_plan = dict(DEFAULT_PROP_PLAN)
    else:
        user_areas_plan = load_json_or_path(args.areas_plan)
        areas_plan = resolve_areas_plan(user_areas_plan, district_layout, rng)

        user_building_plan = load_json_or_path(args.building_plan)
        building_plan = resolve_building_plan(
            user_building_plan, unified_template_manager, rng
        )

        robot_plan = resolve_robot_plan(args.robot_plan)

        user_prop_plan = load_json_or_path(args.prop_plan)
        prop_plan = resolve_prop_plan(user_prop_plan, unified_template_manager, rng)

    # 3) build scene
    builder = SceneGraphBuilder(unified_template_manager, rng_seed=args.seed)
    builder.generate_districts_from_layout(district_layout)
    builder.generate_major_areas(areas_plan)
    builder.generate_primary_roads()
    builder.generate_bridges()

    success = builder.populate_buildings_along_streets(
        building_plan,
        align_offset=args.align_offset,
        road_clearance=args.road_clearance,
    )

    if not success:
        print(f"[Skip] robot_base not placed. Skip saving for: {args.output_prefix}")
        return False

    builder.generate_props_and_robots(robot_plan=robot_plan, prop_plan=prop_plan)

    # optional: apply visual noise to areas
    perlin_info = None
    if args.noise_post:
        perlin_info = apply_perlin_to_areas_consistent(
            builder.nodes,
            seed=args.seed,
            types=("water_body", "garden"),
            amplitude=args.noise_amp,
            scale=args.noise_scale,
            octaves=args.noise_octaves,
            max_seg_len=args.noise_maxseg,
        )

    graph = {"nodes": builder.nodes, "edges": builder.edges}

    # 4) save json and plot
    json_path = f"{args.output_prefix}/scene_graph.json"
    png_path = f"{args.output_prefix}/scene.png"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Saved graph JSON -> {json_path}")

    draw_scene(
        builder.nodes,
        builder.edges,
        out_png=png_path,
        show=args.show,
        perlin_edge_curves=perlin_info["edge_curves"] if perlin_info else None,
        perlin_edge_curves_loose=(
            perlin_info["edge_curves_loose"] if perlin_info else None
        ),
        street_match_quant=perlin_info["street_match_quant"] if perlin_info else 0.5,
    )
    print(f"Saved preview PNG -> {png_path}")

    # 5) summary
    print_summary(builder.nodes, builder.edges)

    # 6) Save snapshot of plans used for reproducibility
    plans_used = {
        "areas_plan": areas_plan,
        "building_plan": building_plan,
        "robot_plan": robot_plan,
        "prop_plan": prop_plan,
        "seed": args.seed,
    }
    with open(f"{args.output_prefix}/plans.json", "w", encoding="utf-8") as f:
        json.dump(plans_used, f, ensure_ascii=False, indent=2)
    print(f"Saved plans snapshot -> {args.output_prefix}/plans.json")


# --------------------------------
# Dataset runner (multi-run to directory)
# --------------------------------
def run_dataset(args):
    root = Path(args.dataset_dir)
    ensure_dir(root)

    for idx in range(args.num_scenarios):
        tag = f"scenario_{idx+1}"
        out_dir = root / tag
        ensure_dir(out_dir)

        # Each scenario gets an independent seed
        args_this = argparse.Namespace(**vars(args))  # Copy args
        args_this.seed = args.seed + idx

        args_this.output_prefix = str(out_dir)
        print(f"\n=== Generating {tag} with seed={args_this.seed} ===")
        res = main(args_this)
        if res is False:
            continue


# --------------------------------
# CLI
# --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate city scenes (random / user-defined four plan types)"
    )

    # Batch run configuration
    parser.add_argument("--num-scenarios", type=int, default=1, help="Number of scenarios to generate")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default= os.path.join(os.path.dirname(__file__), "../../dataset/semantic/scenarios/cybertown"),
        help="Output directory root path",
    )

    # Single run visualization/layout/seed etc.
    parser.add_argument(
        "--use-districts",
        type=bool,
        default=False,
        help="Use 3x3 grid layout (default False=single zone)",
    )
    parser.add_argument("--show", type=bool, default=False, help="Show window")
    parser.add_argument(
        "--world-size", type=float, default=1000.0, help="World size in single-zone layout (meters)"
    )
    parser.add_argument(
        "--cell", type=float, default=300.0, help="Side length of each grid cell (meters)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--align_offset", type=float, default=12.0, help="Building center offset from road (meters)"
    )
    parser.add_argument(
        "--road_clearance", type=float, default=4.0, help="Road clearance buffer distance (meters)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="cybertown_scene",
        help="Output prefix (used in single-run mode)",
    )

    # Perlin visual jitter
    parser.add_argument(
        "--noise-post",
        type=bool,
        default=True,
        help="Apply Perlin post-processing to area boundaries (visual only)",
    )
    parser.add_argument("--noise-amp", type=float, default=20, help="Perlin amplitude (meters)")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Perlin scale")
    parser.add_argument(
        "--noise-octaves", type=int, default=1, help="Perlin octaves (requires noise package)"
    )
    parser.add_argument(
        "--noise-maxseg", type=float, default=8.0, help="Edge subdivision length before jitter (meters)"
    )

    # Four plan types: use user-provided if available; otherwise use random generator
    parser.add_argument(
        "--use-default-plans",
        type=bool,
        default=True,
        help="Use fixed default plans (area/building/robot/prop), ignoring random and user input",
    )
    parser.add_argument(
        "--areas-plan",
        type=str,
        default=None,
        help="Area plan (JSON string or @path.json). Can be: "
        "1) {district_id: {type:count,...}, ...}; "
        "2) {type:count,...} (single plan applied to each district)",
    )
    parser.add_argument(
        "--building-plan",
        type=str,
        default=None,
        help="Building plan (JSON string or @path.json)",
    )
    parser.add_argument(
        "--robot-plan",
        type=str,
        default=None,
        help="Robot plan (JSON string or @path.json). If not provided, uses SceneGraphBuilder internal default",
    )
    parser.add_argument(
        "--prop-plan",
        type=str,
        default=None,
        help="Prop plan (JSON string or @path.json)",
    )

    args = parser.parse_args()

    # Batch or single run
    if args.num_scenarios > 1 or args.dataset_dir:
        run_dataset(args)
    else:
        main(args)


# Random all (single run):
# python ./modules/scenario_builder/generate_scenarios.py

# Use default plans (single run):
# python ./modules/scenario_builder/generate_scenarios.py --use-default-plans True

# User-defined areas:
# python ./modules/scenario_builder/generate_scenarios.py --areas-plan '{"water_body":2,"garden":2,"neighborhood":3}'

# Batch generate 50 scenarios to directory:
# python ./modules/scenario_builder/generate_scenarios.py --num-scenarios 50
