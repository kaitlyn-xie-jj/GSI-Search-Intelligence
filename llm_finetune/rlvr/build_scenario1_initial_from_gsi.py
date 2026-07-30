#!/usr/bin/env python3
"""Build scenario_1 first-plan RLVR rows from the current GSI dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from llm_finetune.rlvr.goal_type_sampling import allocate_goal_type_counts, parse_goal_type_weights_arg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GSI_ROOT = REPO_ROOT
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "rlvr_gsi" / "scenario1_initial_replan_50_50"
PARQUET_COLUMNS = ["prompt", "data_source", "ability", "reward_model", "extra_info"]
SPLITS = ("train", "val", "test")
ENV_START = "Environment observation:"
ENV_END = "\n\n### 2. Robot Skill Library"
ROBOTS_START = "### Current Available Robots\n"
ROBOTS_END = "\n\n### User Instruction"
CYBERTOWN_SCENARIO1_RE = re.compile(r"^cybertown_scenario_1_(g_\d+)$")


@dataclass(frozen=True)
class PromptBase:
    scene_desc: str
    available_robots: str


def _ensure_gsi_import_path(gsi_root: Path = DEFAULT_GSI_ROOT) -> None:
    if str(gsi_root) not in sys.path:
        sys.path.insert(0, str(gsi_root))


def _load_json_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _normal_prompt_cell(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return [
            {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
            for item in prompt
            if isinstance(item, dict)
        ]
    if hasattr(prompt, "tolist"):
        return _normal_prompt_cell(prompt.tolist())
    return [{"role": "user", "content": str(prompt or "")}]


def _prompt_content(row: pd.Series) -> str:
    prompt = _normal_prompt_cell(row.get("prompt"))
    if not prompt:
        return ""
    return prompt[0].get("content", "")


def extract_prompt_base(prompt_content: str) -> PromptBase:
    env_start = prompt_content.find(ENV_START)
    env_end = prompt_content.find(ENV_END, env_start)
    if env_start < 0 or env_end < 0:
        raise ValueError("reference prompt does not contain the expected scenario environment block")

    robots_start = prompt_content.find(ROBOTS_START)
    robots_end = prompt_content.find(ROBOTS_END, robots_start)
    if robots_start < 0 or robots_end < 0:
        raise ValueError("reference prompt does not contain the expected available robots block")

    return PromptBase(
        scene_desc=prompt_content[env_start:env_end].strip(),
        available_robots=prompt_content[robots_start + len(ROBOTS_START) : robots_end].strip(),
    )


def load_prompt_base(reference_dir: Path) -> PromptBase:
    for split in SPLITS:
        path = reference_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            extra = dict(row.get("extra_info") or {})
            if extra.get("rlvr_source", "initial") != "initial":
                continue
            content = _prompt_content(row)
            if content:
                return extract_prompt_base(content)
    raise FileNotFoundError(f"no initial reference prompt found in {reference_dir}")


def collect_excluded_task_ids(exclude_dirs: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for dataset_dir in exclude_dirs:
        for split in SPLITS:
            path = dataset_dir / f"{split}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["extra_info"])
            for extra in df["extra_info"]:
                info = dict(extra or {})
                if info.get("rlvr_source", "initial") != "initial":
                    continue
                task_id = info.get("task_id")
                if task_id:
                    excluded.add(str(task_id))
    return excluded


def _scenario1_goal_ids(tasks_path: Path) -> set[str]:
    goal_ids: set[str] = set()
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {tasks_path}:{line_no}: {exc}") from exc
            if item.get("scenario") != "scenario_1":
                continue
            match = CYBERTOWN_SCENARIO1_RE.match(str(item.get("task_id", "")))
            if match:
                goal_ids.add(match.group(1))
            elif item.get("goal"):
                goal_ids.add(str(item["goal"]))
    return goal_ids


def _iter_candidate_goals(
    *,
    goals_path: Path,
    scenario1_goal_ids: set[str],
    excluded_task_ids: set[str],
    goal_types: set[str] | None,
) -> Iterable[dict[str, Any]]:
    with goals_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {goals_path}:{line_no}: {exc}") from exc

            goal_id = str(item.get("id") or item.get("goal_id") or "")
            if goal_id not in scenario1_goal_ids:
                continue
            task_id = f"scenario_1_{goal_id}"
            if task_id in excluded_task_ids:
                continue

            goal_details = _load_json_maybe(item.get("goal_details"))
            meta = _load_json_maybe(item.get("meta"))
            goal_type = str(goal_details.get("goal_type") or meta.get("goal_type") or "unknown")
            if goal_types and goal_type not in goal_types:
                continue

            instruction = str(item.get("instruction") or goal_details.get("description") or "").strip()
            if not instruction:
                continue
            yield {
                "goal_id": goal_id,
                "task_id": task_id,
                "instruction": instruction,
                "goal_type": goal_type,
                "goal_details": goal_details,
                "meta": meta,
            }


def sample_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_count: int | None,
    balanced: bool,
    goal_type_weights: Mapping[str, float] | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not candidates:
        return []
    if not max_count or max_count >= len(candidates):
        sampled = list(candidates)
        rng.shuffle(sampled)
        return sampled
    if not balanced and not goal_type_weights:
        sampled = list(candidates)
        rng.shuffle(sampled)
        return sampled[:max_count]

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[item["goal_type"]].append(item)
    for items in by_type.values():
        rng.shuffle(items)

    capacities = {goal_type: len(items) for goal_type, items in by_type.items()}
    if balanced and not goal_type_weights:
        goal_type_weights = {goal_type: 1.0 for goal_type in by_type}
    counts = allocate_goal_type_counts(
        total_target=max_count,
        capacities=capacities,
        goal_type_weights=goal_type_weights,
    )

    selected: list[dict[str, Any]] = []
    for goal_type, count in counts.items():
        if count <= 0:
            continue
        selected.extend(by_type[goal_type][:count])
    rng.shuffle(selected)
    return selected


def _build_prompt(*, prompt_base: PromptBase, goal_type: str, instruction: str, gsi_root: Path) -> str:
    _ensure_gsi_import_path(gsi_root)
    from modules.task_solver.sgi_planner.prompt.runtime_builders import compose_master_context, select_prompt_and_feedback

    master_context = compose_master_context(
        planner_mode="full",
        use_environment_model=True,
        scene_desc=prompt_base.scene_desc,
        goal_type=goal_type,
        is_replanning=False,
    )
    template, _ = select_prompt_and_feedback(
        planner_mode="full",
        use_separate_prompts=True,
        is_replanning=False,
    )
    return template.format(
        master_context=master_context,
        available_robots=prompt_base.available_robots,
        instruction=instruction,
    )


def _to_row(candidate: dict[str, Any], *, prompt_base: PromptBase, data_tag: str, gsi_root: Path) -> dict[str, Any]:
    return {
        "prompt": [
            {
                "role": "user",
                "content": _build_prompt(
                    prompt_base=prompt_base,
                    goal_type=candidate["goal_type"],
                    instruction=candidate["instruction"],
                    gsi_root=gsi_root,
                ),
            }
        ],
        "data_source": "cybertown",
        "ability": "planning",
        "reward_model": {"style": "custom", "ground_truth": ""},
        "extra_info": {
            "task_id": candidate["task_id"],
            "scenario_id": "scenario_1",
            "goal_id": candidate["goal_id"],
            "goal_type": candidate["goal_type"],
            "rlvr_source": "initial",
            "data_tag": data_tag,
            "state_store": None,
            "state_id": None,
        },
    }


def _split_rows(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_test = int(round(total * test_ratio))
    n_val = int(round(total * val_ratio))
    n_train = total - n_val - n_test
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _resolve_dataset_file(dataset_root: Path, data_type: str, filename: str) -> Path:
    legacy = dataset_root / data_type / "cybertown" / filename
    if legacy.exists():
        return legacy
    semantic = dataset_root / "semantic" / data_type / "cybertown" / filename
    return semantic


def build_dataset(
    *,
    dataset_root: Path,
    gsi_root: Path,
    output_dir: Path,
    prompt_base: PromptBase,
    excluded_task_ids: set[str],
    max_count: int | None,
    balanced: bool,
    goal_types: set[str] | None,
    goal_type_weights: Mapping[str, float] | None = None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    tasks_path = _resolve_dataset_file(dataset_root, "tasks", "tasks.jsonl")
    goals_path = _resolve_dataset_file(dataset_root, "goals", "goals.jsonl")
    if not tasks_path.exists():
        raise FileNotFoundError(f"missing GSI tasks file: {tasks_path}")
    if not goals_path.exists():
        raise FileNotFoundError(f"missing GSI goals file: {goals_path}")

    scenario1_goal_ids = _scenario1_goal_ids(tasks_path)
    candidates = list(
        _iter_candidate_goals(
            goals_path=goals_path,
            scenario1_goal_ids=scenario1_goal_ids,
            excluded_task_ids=excluded_task_ids,
            goal_types=goal_types,
        )
    )
    sampled = sample_candidates(
        candidates,
        max_count=max_count,
        balanced=balanced,
        goal_type_weights=goal_type_weights,
        seed=seed,
    )
    if not sampled:
        raise ValueError("no non-duplicate scenario_1 candidates found")

    data_tag = output_dir.name
    rows = [_to_row(candidate, prompt_base=prompt_base, data_tag=data_tag, gsi_root=gsi_root) for candidate in sampled]
    splits = _split_rows(rows, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed + 19)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        for row in split_rows:
            row["extra_info"]["split"] = split
        pd.DataFrame(split_rows, columns=PARQUET_COLUMNS).to_parquet(output_dir / f"{split}.parquet", index=False)

    goal_counts = Counter(item["goal_type"] for item in sampled)
    manifest = {
        "dataset_root": str(dataset_root),
        "gsi_root": str(gsi_root),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "candidate_rows_after_exclusion": len(candidates),
        "excluded_existing_initial_task_ids": len(excluded_task_ids),
        "max_count": max_count,
        "balanced": balanced,
        "goal_types": sorted(goal_types) if goal_types else None,
        "goal_type_weights": dict(goal_type_weights) if goal_type_weights else None,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "splits": {split: len(split_rows) for split, split_rows in splits.items()},
        "goal_type_counts": dict(sorted(goal_counts.items())),
        "schema": PARQUET_COLUMNS,
        "invariant": "all rows are initial planning rows for scenario_1 with no state_store/state_id",
        "dedupe_rule": "excluded initial task_ids from --exclude-dir are never emitted",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the GSI dataset root. Supports both dataset/{data_type}/cybertown and dataset/semantic/{data_type}/cybertown.",
    )
    parser.add_argument("--gsi-root", default=str(DEFAULT_GSI_ROOT), help="Path to the GSI repository root.")
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_DIR),
        help="Existing scenario_1 initial dataset used only to extract the base environment prompt block.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Existing RLVR parquet dataset dir whose initial task_ids should be excluded. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val/test parquet files.")
    parser.add_argument("--max-count", type=int, default=10000, help="Maximum number of new initial rows to emit.")
    parser.add_argument("--balanced", action="store_true", help="Round-robin sample across goal_type buckets.")
    parser.add_argument(
        "--goal-types",
        default="",
        help="Optional comma-separated goal_type allowlist, e.g. assembly,transport,patrol.",
    )
    parser.add_argument(
        "--goal-type-weights",
        default="",
        help="Optional JSON string or JSON file path mapping goal_type to sampling weight.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260506)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    exclude_dirs = [Path(item) for item in args.exclude_dir]
    if reference_dir not in exclude_dirs:
        exclude_dirs.append(reference_dir)

    prompt_base = load_prompt_base(reference_dir)
    excluded = collect_excluded_task_ids(exclude_dirs)
    goal_types = {item.strip() for item in args.goal_types.split(",") if item.strip()} or None
    goal_type_weights = parse_goal_type_weights_arg(args.goal_type_weights)
    max_count = args.max_count if args.max_count > 0 else None
    manifest = build_dataset(
        dataset_root=Path(args.dataset_root),
        gsi_root=Path(args.gsi_root),
        output_dir=output_dir,
        prompt_base=prompt_base,
        excluded_task_ids=excluded,
        max_count=max_count,
        balanced=args.balanced,
        goal_types=goal_types,
        goal_type_weights=goal_type_weights,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
