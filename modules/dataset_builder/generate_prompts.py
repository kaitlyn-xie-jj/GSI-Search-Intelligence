#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Generation Script

Generate prompts based on task and scenario data, supporting:
- Loading tasks from task files (tasks.jsonl)
- Loading scenarios from scenario files
- Generating segmented structured prompt data
- Saving as JSONL + dictionary pools using pure Python deduplication
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Project root path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_builder.prompt_utils.prompt_generator import DatasetPromptGenerator
from modules.dataset_builder.prompt_utils.text_deduplicator import (
    DeduplicatedJsonlBuilder,
)

# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

DEFAULT_PLANNER_MODE = "full"
DEFAULT_USE_ENVIRONMENT_MODEL = True

# Default paths
TASKS_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset/semantic/tasks/cybertown/tasks.jsonl",
)
SCENARIOS_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset/semantic/scenarios/cybertown",
)
GOALS_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset/semantic/goals/cybertown/goals.jsonl",
)
PROMPTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset/semantic/prompts/cybertown",
)


# ---------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------


def load_task(task_file: Path) -> Dict[str, Any]:
    """Load a single task file"""
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_goals(goals_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load all goals from JSONL file, indexed by goal_id"""
    if not goals_file.exists():
        raise FileNotFoundError(f"Goals file not found: {goals_file}")

    goals_dict = {}
    with open(goals_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                goal = json.loads(line)
                # Deserialize goal_details
                if "goal_details" in goal and isinstance(goal["goal_details"], str):
                    try:
                        goal["goal_details"] = json.loads(goal["goal_details"])
                    except json.JSONDecodeError:
                        continue
                # Extract goal_id
                goal_id = (
                        goal.get("id")
                        or (goal.get("goal_details") or {}).get("goal_id")
                        or (goal.get("goal") or {}).get("id")
                )
                if goal_id:
                    goals_dict[goal_id] = goal
            except (json.JSONDecodeError, Exception):
                continue
    return goals_dict


def load_tasks_jsonl(
        tasks_file: Path,
        goals_dict: Dict[str, Dict[str, Any]],
        scenarios_dict: Dict[str, Dict[str, Any]],
        type_name: str = "cybertown",
) -> List[Dict[str, Any]]:
    """Load task references from JSONL file and combine into complete task data"""
    if not tasks_file.exists():
        raise FileNotFoundError(f"Task file not found: {tasks_file}")

    tasks = []
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                task_ref = json.loads(line)
                scenario_id = task_ref.get("scenario") or task_ref.get("scenario_id")
                goal_id = task_ref.get("goal") or task_ref.get("goal_id")

                if not scenario_id or not goal_id:
                    continue

                goal_data = goals_dict.get(goal_id)
                scenario_data = scenarios_dict.get(scenario_id)
                if not goal_data or not scenario_data:
                    continue

                tasks.append(
                    {
                        "task_id": f"{type_name}_{scenario_id}_{goal_id}",
                        "scenario": scenario_id,
                        "goal": goal_data,
                        "type": type_name,
                    }
                )
            except (json.JSONDecodeError, Exception):
                continue
    return tasks


def load_tasks(
        tasks_path: Path,
        goals_dict: Dict[str, Dict[str, Any]],
        scenarios_dict: Dict[str, Dict[str, Any]],
        type_name: str = "cybertown",
        pattern: str = "task_*.json",
) -> List[Dict[str, Any]]:
    """Batch load task files (supports directory or JSONL file)"""
    if tasks_path.is_file():
        return load_tasks_jsonl(tasks_path, goals_dict, scenarios_dict, type_name)

    if not tasks_path.is_dir():
        raise ValueError(f"Path is neither a file nor a directory: {tasks_path}")

    tasks = []
    for task_file in sorted(tasks_path.glob(pattern)):
        try:
            task = load_task(task_file)
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load task {task_file}: {e}")
            continue

    return tasks


def load_scenarios(scenarios_dir: Path, scenario_id: str = None) -> Dict[str, Dict[str, Any]]:
    """
    Batch load all scenarios, or load a specific scenario by ID.

    Args:
        scenarios_dir: Scenarios root directory
        scenario_id: Specific scenario ID (optional). If provided, only loads that scenario.

    Returns:
        Dict: Dictionary with scenario IDs as keys and scenario data as values.
    """
    scenarios = {}  # Initialize at outermost scope

    if scenario_id is not None:
        # --- Case 1: Load specified scenario ---
        scenario_dir = scenarios_dir / scenario_id
        scenario_file = scenario_dir / "scene_graph.json"

        if scenario_file.exists():
            try:
                with open(scenario_file, "r", encoding="utf-8") as f:
                    scenarios[scenario_id] = json.load(f)
            except Exception as e:
                print(f"Error loading scenario '{scenario_id}': {e}")
        else:
            # If specified ID does not exist, print warning or ignore
            print(f"Warning: Scenario file not found for ID: {scenario_id}")

    else:
        # --- Case 2: Batch load all scenarios ---
        if scenarios_dir.exists():
            for scenario_dir in scenarios_dir.iterdir():
                if scenario_dir.is_dir():
                    scenario_file = scenario_dir / "scene_graph.json"
                    if scenario_file.exists():
                        try:
                            with open(scenario_file, "r", encoding="utf-8") as f:
                                scenarios[scenario_dir.name] = json.load(f)
                        except Exception as e:
                            # Usually skip errors during batch loading, but printing helps debug
                            print(f"Skipping {scenario_dir.name} due to error: {e}")
                            continue

    return scenarios


# ---------------------------------------------------------------------
# Prompt generation functions
# ---------------------------------------------------------------------


def extract_goal_id(goal_data: Dict[str, Any]) -> str:
    """Extract goal_id from goal data"""
    if isinstance(goal_data, dict):
        if "goal_details" in goal_data and isinstance(goal_data["goal_details"], dict):
            gid = goal_data["goal_details"].get("goal_id")
            if gid:
                return gid
        if "id" in goal_data:
            return goal_data["id"]
    return "unknown"


def generate_prompt_for_task(
        prompt_generator: DatasetPromptGenerator,
        task: Dict[str, Any],
        scenarios: Dict[str, Dict[str, Any]],
        type_name: str = "cybertown",
) -> Optional[Dict[str, Any]]:
    """Generate prompt for a single task"""
    try:
        scenario_id = task.get("scenario")
        scenario_data = scenarios.get(scenario_id)
        if not scenario_data:
            return None

        goal_data = task.get("goal", {})
        goal_id = extract_goal_id(goal_data)

        # Generate segmented data
        segments_data = prompt_generator.generate_prompt_segments(
            task_data=task,
            scenario_data=scenario_data,
            type_name=type_name,
            scenario_id=scenario_id,
            goal_id=goal_id,
        )

        # Build full prompt
        full_prompt = prompt_generator.build_full_prompt(segments_data)

        prompt_data = {
            "task_id": task.get("task_id", f"{type_name}:{scenario_id}:{goal_id}"),
            "task": task,
            "segments": segments_data["segments"],
            "template_info": segments_data["template_info"],
            "metadata": segments_data["metadata"],
            "full_prompt": full_prompt,
        }
        return prompt_data

    except Exception as e:
        print(f"Warning: Failed to generate prompt {task.get('task_id')}: {e}")
        return None


def generate_prompts_with_deduplication(
        tasks: List[Dict[str, Any]],
        scenarios: Dict[str, Dict[str, Any]],
        prompt_generator: DatasetPromptGenerator,
        type_name: str = "cybertown",
) -> DeduplicatedJsonlBuilder:
    """
    Batch generate prompts with deduplication
    Returns: DeduplicatedJsonlBuilder object (containing data)
    """
    print("🚀 Building Prompt dataset with deduplication...")

    # Extract global config (shared by all tasks)
    global_config = {
        "planner_mode": prompt_generator.planner_mode,
        "use_environment_model": prompt_generator.use_environment_model,
        "template_planner_mode": prompt_generator.planner_mode,
    }

    # Initialize pure Python deduplication builder
    builder = DeduplicatedJsonlBuilder(
        deduplicated_fields={
            "skill_set": "master_skill_set",
            "env_desc": "master_env_description",
            "goal_notes": "master_goal_type_notes",
            "core_def": "master_core_definitions",
            "univ_rules": "master_universal_rules",
            "available_robots": "available_robots",
            "response_format": "response_format",  # Register FORMAT component to dedup pool
            "head_template": "prompt_head_template",  # Deduplicate head template too (same for all records)
        },
        direct_fields=[
            "instruction",
            "feedback_context",
        ],
        global_config=global_config,
    )

    total = len(tasks)
    for idx, task in enumerate(tasks):
        try:
            scenario_id = task.get("scenario")
            scenario_data = scenarios.get(scenario_id)
            if not scenario_data:
                continue

            goal_data = task.get("goal", {})
            goal_id = extract_goal_id(goal_data)

            # Generate segments
            segments_data = prompt_generator.generate_prompt_segments(
                task_data=task,
                scenario_data=scenario_data,
                type_name=type_name,
                scenario_id=scenario_id,
                goal_id=goal_id,
            )

            segments = segments_data.get("segments", {})
            template_info = segments_data.get("template_info", {})
            metadata = segments_data.get("metadata", {})

            # Pass metadata and other fields via kwargs
            # Note: global config fields are automatically filtered
            # Head template and Format are stored via dedup pool (already handled in segments)
            # Ensure task_id is at top level (passed as kwargs, added directly to top level)
            task_id_value = task.get("task_id")
            task_type_value = task.get("type", type_name)
            builder.add_record(
                segments=segments,
                metadata={
                    **metadata,  # Contains task-specific fields like goal_id, scenario_id, goal_type
                },
                task_id=task_id_value,  # Added directly as top-level field for quick lookup
                type=task_type_value,
            )

            if (idx + 1) % 1000 == 0:
                print(f"  ...processed {idx + 1}/{total} tasks")

        except Exception as e:
            print(f"Warning: Task processing error: {e}")
            continue

    return builder


def generate_prompts_without_deduplication(
        tasks: List[Dict[str, Any]],
        scenarios: Dict[str, Dict[str, Any]],
        prompt_generator: DatasetPromptGenerator,
        type_name: str = "cybertown",
) -> List[Dict[str, Any]]:
    """Generate plain list format"""
    prompts = []
    total = len(tasks)

    for idx, task in enumerate(tasks):
        prompt_data = generate_prompt_for_task(
            prompt_generator, task, scenarios, type_name
        )
        if prompt_data:
            # Clean up data, keep only generated prompt text and IDs to reduce size
            simplified = {
                "task_id": prompt_data["task_id"],
                "type": type_name,
                "segments": prompt_data["segments"],
                "prompt": prompt_data["full_prompt"],
                "metadata": prompt_data["metadata"],
            }
            prompts.append(simplified)

        if (idx + 1) % 10000 == 0:
            print(f"  ...generated {idx + 1}/{total}")

    return prompts


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Prompt Dataset (Pure Python)")

    # Input
    parser.add_argument("--tasks-dir", type=str, default=TASKS_DIR_DEFAULT)
    parser.add_argument("--scenarios-dir", type=str, default=SCENARIOS_DIR_DEFAULT)
    parser.add_argument("--scenario-id", type=str, default='scenario_1')

    parser.add_argument("--goals-dir", type=str, default=GOALS_DIR_DEFAULT)

    # Output
    parser.add_argument("--output-dir", type=str, default=PROMPTS_DIR_DEFAULT)
    parser.add_argument(
        "--disable-deduplication",
        action="store_true",
        help="Disable deduplication (generates a huge single JSONL file)",
    )

    # Configuration
    parser.add_argument("--planner-mode", type=str, default=DEFAULT_PLANNER_MODE)
    parser.add_argument("--type-name", type=str, default="cybertown")

    args = parser.parse_args()

    # 1. Prepare generator
    prompt_generator = DatasetPromptGenerator(
        planner_mode=args.planner_mode,
        use_environment_model=True,
    )

    # 2. Load data
    scenarios_path = Path(args.scenarios_dir)
    goals_path = Path(args.goals_dir)
    tasks_path = Path(args.tasks_dir)

    print("=== Data Loading Phase ===")
    scenarios = load_scenarios(scenarios_path,scenario_id=args.scenario_id)
    print(f"✅ Scenarios: {len(scenarios)}")

    goals_dict = load_goals(goals_path)
    print(f"✅ Goals: {len(goals_dict)}")

    tasks = load_tasks(tasks_path, goals_dict, scenarios, args.type_name)
    print(f"✅ Tasks: {len(tasks)}")

    if not tasks:
        print("❌ No valid tasks found, exiting")
        return 1

    # 3. Generate Prompts
    print("\n=== Prompt Generation Phase ===")
    output_dir = Path(args.output_dir)

    if not args.disable_deduplication:
        # --- Deduplication mode (recommended) ---
        builder = generate_prompts_with_deduplication(
            tasks, scenarios, prompt_generator, args.type_name
        )

        # Get statistics and save
        stats = builder.get_stats()
        print(f"\n✅ Generation complete: {stats['total_records']} records")

        print(f"💾 Saving data to: {output_dir}")
        builder.save(str(output_dir), main_filename="prompts.jsonl")

        print("\nDeduplication statistics:")
        for pool_name, info in stats["pools"].items():
            print(f"  - {pool_name}: {info['unique_texts']} unique items")

    else:
        # --- Normal mode (files will be large) ---
        print("⚠️  Deduplication disabled, generating full data...")
        prompts = generate_prompts_without_deduplication(
            tasks, scenarios, prompt_generator, args.type_name
        )

        output_file = output_dir / "prompts_full.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving data to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        # Build index (normal mode)
        index_file = output_file.with_name(output_file.name + ".index")
        print(f"Building index file: {index_file} ...")
        index = {}
        with open(output_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    task_id = record.get("task_id")
                    if task_id:
                        index[str(task_id)] = offset
                except json.JSONDecodeError:
                    continue

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f)

        print(f"✅ Saved {len(prompts)} records")

    return 0


if __name__ == "__main__":
    exit(main())
