# run/utils/batch_runner.py
import asyncio
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime

import datasets
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any, NoReturn, Optional, Tuple
from modules.config.system_config import config

from run.utils.analysis import aggregate_experiment_results
from run.utils.case_runner import run_single_case_lifecycle

@dataclass
class ParallelBatchConfig:
    batch_name: Optional[str] = None
    output_root: Path = Path(__file__).resolve().parent.parent.parent / "results/batch_runs"
    max_workers: int = 50  # High concurrency by default; can be set to 100.
    enable_replanning: bool = True
    enable_render: bool = False
    solver_kwargs: Optional[Dict[str, Any]] = None
    config_overrides: Optional[Dict[str, Any]] = None


# Top-level worker function; must stay top-level for pickle.
def _worker_entry(kwargs):
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    datasets.disable_progress_bar()

    # Apply config overrides before creating the solver.
    config_overrides = kwargs.get('config_overrides') or {}
    if config_overrides:
        for k, v in config_overrides.items():
            config.set_override(k, v)

    # Unpack parameters and call lifecycle.
    return asyncio.run(run_single_case_lifecycle(
        run_dir=kwargs['run_dir'],
        task_id=kwargs['task_id'],
        solver_kwargs=kwargs['solver_kwargs'],
        enable_replanning=kwargs['enable_replanning'],
        enable_render=kwargs['enable_render'],
    ))


class ParallelBatchRunner:
    def __init__(self, cfg: ParallelBatchConfig):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.batch_root = cfg.output_root / (cfg.batch_name or f"batch_{ts}")
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

    def run_many(self, task_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Run multiple tasks concurrently
        Args:
            task_ids: Task ID list ["cybertown/s1/g1", ...]
        """
        if not task_ids:
            print("No tasks to run.")
            return []
        # Control concurrency. Up to 100 is fine, but machine resources still matter.
        max_workers = max(1, min(self.cfg.max_workers, len(task_ids)))

        tasks_args = []
        for i, tid in enumerate(task_ids, 1):
            # Replace / with _ to avoid nested directories and keep the structure flat.
            run_dir = self.batch_root / f"{i:04d}_{tid}"
            tasks_args.append(
                {
                    "run_dir": str(run_dir),
                    "task_id": tid,
                    "enable_replanning": self.cfg.enable_replanning,
                    "enable_render": self.cfg.enable_render,
                    "solver_kwargs": self.cfg.solver_kwargs or {},
                    "config_overrides": self.cfg.config_overrides or {},
                }
            )

        results: List[Dict[str, Any]] = []
        running_futures = set()

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            with tqdm(total=len(tasks_args), desc="Running Batch", unit="task") as pbar:

                # Use an iterator for streaming task processing.
                task_iter = iter(tasks_args)

                while True:
                    # 1. Fill the task pool while it has room and tasks remain.
                    #    Keep max_workers * 2 tasks in the pool to keep CPUs busy without exhausting memory.
                    while len(running_futures) < max_workers * 2:
                        try:
                            arg = next(task_iter)
                            fut = pool.submit(_worker_entry, arg)
                            fut.arg_data = arg
                            running_futures.add(fut)
                        except StopIteration:
                            break  # All tasks have been submitted.

                    # 2. If no tasks are running, all work is done.
                    if not running_futures:
                        break

                    # 3. Wait until at least one task completes.
                    #    return_when=FIRST_COMPLETED is key for streaming processing.
                    done, _ = wait(running_futures, return_when=FIRST_COMPLETED)

                    # 4. Process completed tasks.
                    for fut in done:
                        running_futures.remove(fut)  # Remove from the running set.
                        try:
                            res = fut.result()
                        except Exception as e:
                            traceback.print_exc()
                            # Retrieve parameters from the attribute we attached.
                            arg = fut.arg_data
                            res = {
                                "ok": False,
                                "error": str(e),
                                "task_id": arg['task_id'],
                                "run_dir": str(arg['run_dir'])
                            }
                        results.append(res)
                        pbar.update(1)  # Update the progress bar.

        self._save_results(results)
        return results

    def _save_results(self, results):
        """
        Aggregate results, inject parameters, and print to console
        """
        # 1. Save per-case results (summary.jsonl).
        summary_path = self.batch_root / "summary.jsonl"
        with open(summary_path, 'w', encoding='utf-8') as f:
            for r in results:
                # default=str handles non-serializable objects.
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
        print(f"[ParallelBatchRunner] Batch completed: {summary_path}")

        # 2. Compute aggregate statistics.
        agg = aggregate_experiment_results(results)

        # 3. Inject input parameters.
        skw = dict(self.cfg.solver_kwargs or {})
        input_params = {
            "planner_mode": skw.get("planner_mode") or (config.planner_mode or "phase"),
            "robot_type_list": skw.get("robot_type_list")
                               or (
                                       config.default_robot_types
                                       or ["UAV", "UGV", "Quadruped", "Humanoid"]
                               ),
            "enable_render": bool(self.cfg.enable_render),
            "max_workers": self.cfg.max_workers,
            "config_overrides": self.cfg.config_overrides or {},
        }
        agg["input_params"] = input_params

        # 4. Print statistics to console.
        print(
            "[ParallelBatchRunner] Overall statistics:",
            json.dumps(agg.get("overall", {}), ensure_ascii=False, indent=2),
        )
        print(
            "[ParallelBatchRunner] Statistics by goal type:",
            json.dumps(agg.get("by_goal_type", {}), ensure_ascii=False, indent=2),
        )

        # 5. Save aggregate results (aggregate_full.json).
        agg_path = self.batch_root / "aggregate_full.json"
        t = time.perf_counter()
        with open(agg_path, 'w', encoding='utf-8') as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)

        print(f"[Batch] Aggregate statistics saved to: {agg_path}")
