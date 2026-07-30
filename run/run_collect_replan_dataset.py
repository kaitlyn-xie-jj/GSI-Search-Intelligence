# run_collect_replan_dataset.py
import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_loader import DatasetLoader
from modules.utils.process.dataset_visulizer import get_and_print_distribution


from run.utils.batch_runner import ParallelBatchRunner, ParallelBatchConfig
from run.utils.common import discover_tasks_from_args
from run.utils.analysis import merge_replan_records
from modules.config.system_config import config
from modules.utils.replan_recorder import ReplanDatasetRecorder


def resolve_dataset_root(dataset_root_arg: str | None = None) -> Path:
    """Resolve the local GSI dataset root used by task discovery and workers."""
    root = dataset_root_arg or os.environ.get("GSI_DATASET_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (project_root / "dataset").resolve()


def _split_csv_values(values):
    result = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch collect replanning dataset")

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
    filter_grp.add_argument("--max-count", type=int, default=100, help="Max task count")

    # === 2. Runtime Config ===
    run_grp = parser.add_argument_group("Runtime Config")
    run_grp.add_argument("--batch-name", type=str, default=None)
    run_grp.add_argument("--output-root", type=str,
                         default=Path(__file__).resolve().parent.parent / "results/replan_batch_runs")
    run_grp.add_argument("--max-workers", type=int, default=10)
    run_grp.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Local GSI dataset root. Defaults to $GSI_DATASET_ROOT or <repo>/dataset.",
    )
    run_grp.add_argument(
        "--max-newcases-per-run",
        type=int,
        default=1,
        help="Maximum number of new situations injected per task; must be greater than 0 when collecting replan samples",
    )
    run_grp.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Task sampling random seed, used to avoid repeated samples from a fixed seed during additional collection",
    )
    capture_grp = parser.add_argument_group("Replan Capture Filters")
    capture_grp.add_argument(
        "--capture-min-newcases",
        type=int,
        default=1,
        help="Minimum number of new cases before collecting a full replan prompt",
    )
    capture_grp.add_argument(
        "--capture-min-replan-index",
        type=int,
        default=1,
        help="Minimum full replan prompt index to collect; useful for skipping T1 or early replans",
    )
    capture_grp.add_argument(
        "--capture-goal-types",
        nargs="*",
        default=None,
        help="Collect only these goal_type values; supports space or comma separators",
    )
    capture_grp.add_argument(
        "--capture-event-types",
        nargs="*",
        default=None,
        help="Collect only these event types; supports space or comma separators",
    )
    capture_grp.add_argument(
        "--capture-event-reasons",
        nargs="*",
        default=None,
        help="Collect only these event reasons; supports space or comma separators",
    )
    capture_grp.add_argument(
        "--capture-max-records-per-run",
        type=int,
        default=1,
        help="Maximum number of replan_records entries to write per run",
    )
    capture_grp.add_argument(
        "--capture-stop-after-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to end the current run right after collecting a sample; use --no-capture-stop-after-record to continue later replans",
    )
    capture_grp.add_argument(
        "--capture-save-llm-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save each TaskPlan LLM prompt/response; enabled by default and writes after the LLM returns for matched replan samples",
    )
    capture_grp.add_argument(
        "--capture-data-tag",
        type=str,
        default=None,
        help="data_tag written to replan_records.jsonl/llm_trace.jsonl for filtering batches later",
    )
    capture_grp.add_argument(
        "--capture-require-prompt-event-match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only replan samples whose prompt feedback matches a newcase snapshot",
    )
    capture_grp.add_argument(
        "--capture-single-event-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only replan samples whose prompt feedback contains exactly one newcase event",
    )

    args = parser.parse_args()

    capture_options = {
        "min_newcases": args.capture_min_newcases,
        "min_replan_index": args.capture_min_replan_index,
        "goal_types": _split_csv_values(args.capture_goal_types),
        "event_types": _split_csv_values(args.capture_event_types),
        "event_reasons": _split_csv_values(args.capture_event_reasons),
        "max_records_per_run": args.capture_max_records_per_run,
        "stop_after_record": args.capture_stop_after_record,
        "save_llm_io": args.capture_save_llm_io,
        "data_tag": args.capture_data_tag or args.batch_name,
        "require_prompt_event_match": args.capture_require_prompt_event_match,
        "single_event_only": args.capture_single_event_only,
    }

    # Enable the parent process for any direct, non-worker use; worker processes
    # receive the same options through solver_kwargs below.
    ReplanDatasetRecorder.enable(**capture_options)

    # 1. Discover tasks.
    print("=== Replan Collection Discovery ===")
    dataset_root = resolve_dataset_root(args.dataset_root)
    os.environ["GSI_DATASET_ROOT"] = str(dataset_root)
    print(f"Dataset root: {dataset_root}")
    loader = DatasetLoader(
        local_path=str(dataset_root),
        use_local=True,
    )

    task_ids = discover_tasks_from_args(loader, args)
    get_and_print_distribution(loader, task_ids)
    print(f"Tasks to process: {len(task_ids)}")

    if not task_ids:
        sys.exit(0)

    # 2. Configure, forcing replanning and collection on.
    cfg = ParallelBatchConfig(
        batch_name=args.batch_name,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        enable_replanning=True,  # Collection requires replanning.
        enable_render=False,  # Collection usually does not need rendering.
        solver_kwargs={
            "enable_replan_dataset_capture": True,
            "planner_mode": config.planner_mode or "full",
            "replan_capture_options": capture_options,
        },
        config_overrides={
            "enable_new_case_generation": True,
            "enable_replanning": True,
            "max_newcases_per_run": args.max_newcases_per_run,
        },
    )

    # 3. Run.
    runner = ParallelBatchRunner(cfg)
    results = runner.run_many(task_ids)

    # 4. Merge collected results.
    merge_replan_records(runner.batch_root, results)


if __name__ == "__main__":
    main()
