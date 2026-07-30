# run_exp.py
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from run.utils.case_runner import run_single_case_lifecycle
from modules.config.system_config import config


async def main():
    parser = argparse.ArgumentParser(description="SGI Single Task Experiment Runner (Single Case)")

    # Single-task mode accepts exactly one task-id.
    parser.add_argument("--task-id", type=str, default="cybertown_scenario_1_g_6798",
                        help="Task ID (format: type_scenario_goal, e.g. cybertown_scenario_1_g_20)")

    parser.add_argument("--save-dir", type=str, default=None, help="Result save path")
    parser.add_argument("--no-replanning", action="store_true", help="Disable replanning")
    parser.add_argument("--renderers", action="store_true", help="Enable renderers (GUI/Vis)")

    args = parser.parse_args()

    # Build save path.
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        # Default: results/single_<timestamp>_<clean_id>.
        safe_name = args.task_id
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = project_root / "results"/ f"single_{ts}_{safe_name}"

    print(f"=== SGI Single Run ===")
    print(f"Task ID : {args.task_id}")
    print(f"Output  : {save_dir}")

    # Call the shared case lifecycle.
    result = await run_single_case_lifecycle(
        run_dir=save_dir,
        task_id=args.task_id,
        enable_replanning=not args.no_replanning,
        enable_render=args.renderers,
        solver_kwargs={
            "planner_mode": config.planner_mode or "full",
            "robot_type_list": config.default_robot_types or ["UAV", "UGV", "Quadruped", "Humanoid"]
        }
    )

    print("\n=== Execution Summary ===")
    print(f"Status  : {'SUCCESS' if result.get('ok') else 'FAILED'}")
    print(f"Elapsed : {result.get('elapsed_sec', 0)}s")
    if not result.get("ok"):
        print(f"Error   : {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
