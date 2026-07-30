#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-method batch comparison test script

Run the same task set sequentially across multiple methods, each method generates
an independent batch directory and aggregated results.
Optionally enable new case injection dimension, iterating over different new case counts.

Usage:
    # Without new cases (default)
    python run/run_exp_multi_method.py

    # Enable new cases, iterate counts 1,2,3,4
    python run/run_exp_multi_method.py --enable-newcase --newcase-counts 1 2 3 4

    # Test specific methods + new cases
    python run/run_exp_multi_method.py --methods lipllm spine sgi --enable-newcase --newcase-counts 1 3

Directory structure:
    Without new cases:
        results/batch_runs/<timestamp>/
            batch_<method>/
                aggregate_full.json
                ...

    With new cases:
        results/batch_runs/<timestamp>/
            newcase_0/
                batch_<method>/          <- new cases disabled (baseline)
            newcase_1/
                batch_<method>/          <- max_newcases_per_run=1
            newcase_2/
                batch_<method>/          <- max_newcases_per_run=2
            ...
"""
import argparse
import os
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_loader import DatasetLoader
from modules.utils.process.dataset_visulizer import get_and_print_distribution
from modules.config.system_config import config

from run.utils.batch_runner import ParallelBatchRunner, ParallelBatchConfig
from run.utils.common import discover_tasks_from_args

ALL_METHODS = ["sgi", "spine", "smartllm", "lipllm"]


def resolve_dataset_root(dataset_root_arg: str | None = None) -> Path:
    """Resolve the local GSI dataset root used by task discovery and workers."""
    root = dataset_root_arg or os.environ.get("GSI_DATASET_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (project_root / "dataset").resolve()


def _run_one_batch(method: str, output_root: Path, task_ids: list, args, config_overrides: dict = None):
    """Run a single batch test for one method."""
    cfg = ParallelBatchConfig(
        batch_name=f"batch_{method}",
        output_root=output_root,
        max_workers=args.max_workers,
        enable_replanning=(not args.no_replanning),
        solver_kwargs={
            "solver_type": method,
            "planner_mode": config.planner_mode or "full",
            "robot_type_list": config.default_robot_types or ["UAV", "UGV", "Quadruped", "Humanoid"],
        },
        config_overrides=config_overrides,
    )
    runner = ParallelBatchRunner(cfg)
    runner.run_many(task_ids)


def main():
    parser = argparse.ArgumentParser(description="Multi-method batch comparison test (with new case dimension support)")

    # === Method selection ===
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS,
        help=f"List of methods to test, default all: {ALL_METHODS}",
    )

    # === New situation dimension ===
    nc_grp = parser.add_argument_group("New Case Config")
    nc_grp.add_argument(
        "--enable-newcase", default=False, action="store_true",
        help="Enable new case injection dimension",
    )
    nc_grp.add_argument(
        "--newcase-counts", nargs="+", type=int, default=[1, 2, 3, 4],
        help="New case count list, default 1 2 3 4",
    )

    # === Task filters ===
    filter_grp = parser.add_argument_group("Task Filters")
    filter_grp.add_argument("--task-mix", dest="task_mix", nargs="+", default=[
        "area_search=0.1", "assembly=0.1", "emergency_response=0.1",
        "evidence_collection=0.1", "guidance=0.1", "patrol=0.1",
        "target_following=0.1", "traffic_enforcement=0.1",
        "transport=0.1", "verbal_broadcast=0.1",
    ])
    filter_grp.add_argument("--plan-level", dest="plan_levels", nargs="*", default=None)
    filter_grp.add_argument("--coor-level", dest="coor_levels", nargs="*", default=None)
    filter_grp.add_argument("--lang-level", dest="lang_levels", nargs="*", default=None)
    filter_grp.add_argument("--max-count", type=int, default=500)

    # === Runtime config ===
    run_grp = parser.add_argument_group("Runtime Config")
    run_grp.add_argument("--output-root", type=str,
                         default=str(project_root / "results" / "batch_runs"))
    run_grp.add_argument("--max-workers", type=int, default=20)
    run_grp.add_argument("--no-replanning", default=False, action="store_true")
    run_grp.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local GSI dataset root. Defaults to $GSI_DATASET_ROOT or <repo>/dataset.",
    )

    args = parser.parse_args()

    # Validate method names.
    for m in args.methods:
        if m not in ALL_METHODS:
            print(f"[ERROR] Unknown method: {m}, supported methods: {ALL_METHODS}")
            sys.exit(1)

    # Initialize random seeds. Fixed seeds ensure all methods use the same task set.
    random.seed(51)
    np.random.seed(51)

    # 1. Discover tasks once, shared by all methods.
    print("=" * 60)
    print("Task Discovery")
    print("=" * 60)
    dataset_root = resolve_dataset_root(args.dataset_root)
    os.environ["GSI_DATASET_ROOT"] = str(dataset_root)
    print(f"Dataset root: {dataset_root}")
    loader = DatasetLoader(
        local_path=str(dataset_root),
        use_local=True,
    )
    task_ids = discover_tasks_from_args(loader, args)
    get_and_print_distribution(loader, task_ids)
    print(f"Total tasks: {len(task_ids)}")

    if not task_ids:
        print("No tasks found. Exiting.")
        sys.exit(0)

    # 2. Build run plan.
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_root = Path(args.output_root) / ts

    if args.enable_newcase:
        # New situation mode: iterate over [0] plus newcase_counts.
        nc_levels = [0] + sorted(set(args.newcase_counts))
        total_runs = len(args.methods) * len(nc_levels)
        run_idx = 0

        for nc in nc_levels:
            nc_dir = session_root / f"newcase_{nc}"
            if nc == 0:
                overrides = {
                    "enable_new_case_generation": False,
                }
            else:
                overrides = {
                    "enable_new_case_generation": True,
                    "enable_replanning": True,
                    "max_newcases_per_run": nc,
                }

            for method in args.methods:
                run_idx += 1
                print()
                print("=" * 60)
                print(f"[{run_idx}/{total_runs}] method={method}, newcase_max={nc}")
                print("=" * 60)
                _run_one_batch(method, nc_dir, task_ids, args, config_overrides=overrides)
    else:
        # No new situation mode.
        total_runs = len(args.methods)
        for i, method in enumerate(args.methods, 1):
            print()
            print("=" * 60)
            print(f"[{i}/{total_runs}] Running method: {method}")
            print("=" * 60)
            _run_one_batch(method, session_root, task_ids, args)

    print()
    print("=" * 60)
    print(f"All runs completed. Results: {session_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
