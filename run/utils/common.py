# run/utils/common.py
import re
import random
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from modules.config.system_config import config
from modules.utils.trace_loader import load_replay_trace, ReplayTrace

try:
    from modules.dataset_loader.loader import DatasetLoader
except ImportError:
    print("Warning: Could not import DatasetLoader from modules.ds_loader.loader")


def setup_project_path():
    """Automatically add the project root directory to sys.path."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


async def reset_global_environment() -> None:
    """Fully reset all global singletons."""
    from modules.events import stop_global_event_bus, set_global_event_bus
    from modules.platform.platform_factory import cleanup_platform, reset_platform
    from modules.monitor.task_monitor import stop_task_monitoring, reset_task_monitoring

    await stop_global_event_bus()
    set_global_event_bus(None)
    await cleanup_platform()
    reset_platform()
    await stop_task_monitoring()
    reset_task_monitoring()


def find_matching_replay_trace(
        task_id: str,
        project_root: Path
) -> Optional[ReplayTrace]:
    """
    Find replay data by task_id in type/scenario/goal form.
    Supports '/', ':', and '_' separators.
    """
    # Parse task_id into (type_name, scenario_id, goal_id).
    if '/' in task_id:
        parts = task_id.split('/')
        if len(parts) != 3:
            return None
        type_name, scenario_id, goal_id = parts
    elif ':' in task_id:
        parts = task_id.split(':')
        if len(parts) != 3:
            return None
        type_name, scenario_id, goal_id = parts
    else:
        match = re.match(r"(.*)_(scenario_.*)_(g_.*)", task_id)
        if match:
            type_name, scenario_id, goal_id = match.groups()
        else:
            return None

    replay_cfg: Dict[str, Any] = getattr(config, "replay_mode", {}) or {}
    if not replay_cfg.get("enabled"):
        return None

    trace_root = replay_cfg.get("trace_root")
    trace_tag = replay_cfg.get("trace_tag", "default")

    if not trace_root:
        return None

    base = Path(trace_root)
    if not base.is_absolute():
        base = (project_root / trace_root).resolve()

    if not base.is_dir():
        return None

    def _try_load_and_match(run_dir: Path) -> Optional[ReplayTrace]:
        """Try loading a trace from run_dir and matching task_id."""
        if not (run_dir / "temp_vars.jsonl").exists():
            return None
        try:
            trace = load_replay_trace(run_dir, tag=trace_tag)
            ri = trace.run_input or {}
            if (str(ri.get("type_name")) == type_name and
                    str(ri.get("scenario_id")) == scenario_id and
                    str(ri.get("goal_id")) == goal_id):
                return trace
        except Exception:
            pass
        return None

    # First check whether base itself is a run directory.
    direct = _try_load_and_match(base)
    if direct is not None:
        return direct

    # Otherwise, iterate through child directories.
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        matched = _try_load_and_match(child)
        if matched is not None:
            return matched

    return None


def discover_tasks_from_args(loader:DatasetLoader, args: argparse.Namespace) -> List[str]:
    """
    Use the existing loader.get_subset method for mixed sampling.
    Logic: compute quotas on the client and call get_subset multiple times.
    """
    try:
        # 1. Prepare global filters applied to all types.
        global_filters = {}
        if hasattr(args, 'plan_levels') and args.plan_levels:
            global_filters["plan_level"] = args.plan_levels
        if hasattr(args, 'coor_levels') and args.coor_levels:
            global_filters["coor_level"] = args.coor_levels
        if hasattr(args, 'lang_levels') and args.lang_levels:
            global_filters["language_level"] = args.lang_levels

        # 2. Determine the task mix list.
        # args.task_mix already has a default full list in run_exp_parallel.py,
        # so this is always a list, such as ['transport=0.8', 'search=0.2'] or ['transport', 'search'].
        mix_config = args.task_mix
        total_limit = getattr(args, 'max_count', 20)
        sample_seed = getattr(args, 'sample_seed', 42)

        # 3. Parse weights and compute quotas.
        parsed_specs = []
        for item in mix_config:
            if "=" in item:
                g_type, w = item.split("=")
                parsed_specs.append({"type": g_type, "weight": float(w)})
            else:
                parsed_specs.append({"type": item, "weight": 0.0})

        total_spec_weight = sum(s["weight"] for s in parsed_specs)
        undefined_count = len([s for s in parsed_specs if s["weight"] <= 0])
        remaining_weight = max(0.0, 1.0 - total_spec_weight)

        default_weight = (remaining_weight / undefined_count) if undefined_count > 0 else 0

        # 4. Call get_subset in a loop.
        all_task_ids = []
        current_total_count = 0

        print(f"Task Mix Plan (Total limit: {total_limit}):")

        for idx, spec in enumerate(parsed_specs):
            # Determine the current type's weight.
            weight = spec["weight"] if spec["weight"] > 0 else default_weight

            # Compute target count.
            target_count = int(total_limit * weight)

            if idx == len(parsed_specs) - 1:
                remainder = total_limit - current_total_count
                # If target_count is less than the remaining need, use the remainder.
                # Usually we want to fill total_limit as much as possible.
                if current_total_count + target_count < total_limit:
                    target_count = remainder

            # Build filters: global filters plus current type.
            current_filters = {"goal_type": spec["type"]}
            current_filters.update(global_filters)

            if target_count > 0:
                subset = loader.get_subset(
                    filters=current_filters,
                    limit=target_count,
                    seed=sample_seed,
                    name=f"mix_{spec['type']}"
                )

                count_got = len(subset)
                all_task_ids.extend(subset)
                current_total_count += count_got

                print(f"  - {spec['type']:<20}: Weight={weight:.2f} | Wanted={target_count} | Got={count_got}")
            else:
                print(f"  - {spec['type']:<20}: Weight={weight:.2f} | Wanted=0 | Skipped")

        # 5. Shuffle all results.
        random.Random(sample_seed).shuffle(all_task_ids)

        return all_task_ids

    except Exception as e:
        print(f"  [Error] Task Discovery Failed: {e}")
        import traceback
        traceback.print_exc()
        return []
