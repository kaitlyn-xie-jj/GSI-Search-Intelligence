# run_exp_parallel.py
import argparse
import sys
from pathlib import Path
import random
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_loader import DatasetLoader
from modules.utils.process.dataset_visulizer import get_and_print_distribution
from run.utils.batch_runner import ParallelBatchRunner, ParallelBatchConfig
from run.utils.common import discover_tasks_from_args
from modules.config.system_config import config


def main():
    parser = argparse.ArgumentParser(description="SGI Parallel Batch Runner")

    # === 1. Task Filters ===
    filter_grp = parser.add_argument_group("Task Filters")

    # Attribute and difficulty filters.
    filter_grp.add_argument("--task-mix",
                            dest="task_mix",
                            nargs="+",  # Allow multiple arguments.
                            default=[
                                "area_search=0.1",
                                "assembly=0.1",
                                "emergency_response=0.1",
                                "evidence_collection=0.1",
                                "guidance=0.1",
                                "patrol=0.1",
                                "target_following=0.1",
                                "traffic_enforcement=0.1",
                                "transport=0.1",
                                "verbal_broadcast=0.1"
                            ],
                            help="Goal type (e.g. transport, search)")
    filter_grp.add_argument("--plan-level", dest="plan_levels", nargs="*", default=None, help="Planning difficulty (e.g. L1 L2)")
    filter_grp.add_argument("--coor-level", dest="coor_levels", nargs="*", default=None, help="Coordination difficulty (e.g. L1)")
    filter_grp.add_argument("--lang-level", dest="lang_levels", nargs="*", default=None, help="Language difficulty (e.g. L0)")

    # Count limit.
    filter_grp.add_argument("--max-count", type=int, default=1000, help="Maximum task count")
    filter_grp.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Task sampling random seed, used to change the parallel evaluation sample set",
    )

    # === 2. Runtime Config ===
    run_grp = parser.add_argument_group("Runtime Config")
    run_grp.add_argument("--batch-name", type=str, default=None)
    run_grp.add_argument("--output-root", type=str,
                         default=Path(__file__).resolve().parent.parent / "results/batch_runs")
    run_grp.add_argument("--max-workers", type=int, default=40)
    run_grp.add_argument("--no-replanning", default=False, action="store_true")
    run_grp.add_argument("--renderers", default=False, action="store_true")

    args = parser.parse_args()

    # Initialize random seeds.
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)

    # 1. Discover tasks.
    print("=== Task Discovery ===")
    loader = DatasetLoader(
        local_path=str(project_root / "dataset"),
        use_local=True,
    )

    task_ids = discover_tasks_from_args(loader, args)
    get_and_print_distribution(loader, task_ids)
    print(f"Total tasks scheduled: {len(task_ids)}")
    if len(task_ids) > 0:
        print(f"Tasks to process: {task_ids}")
    else:
        print("No tasks found matching criteria. Exiting.")
        sys.exit(0)

    if not task_ids:
        print("No tasks found matching criteria. Exiting.")
        sys.exit(0)

    # 2. Configure runner.
    cfg = ParallelBatchConfig(
        batch_name=args.batch_name,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        enable_replanning=(not args.no_replanning),
        solver_kwargs={
            "planner_mode": config.planner_mode or "full",
            "robot_type_list": config.default_robot_types or ["UAV", "UGV", "Quadruped", "Humanoid"]
        }
    )

    # 3. Execute.
    runner = ParallelBatchRunner(cfg)
    runner.run_many(task_ids)


if __name__ == "__main__":
    main()
