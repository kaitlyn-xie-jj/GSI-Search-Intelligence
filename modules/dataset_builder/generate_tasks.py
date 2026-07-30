"""
Task Generator for Multi-Robot Planning System

Batch generate task files by combining scenarios and goals.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TaskGenerator:
    """Task generator - combines scenarios and goals to generate task files"""

    def __init__(self, dataset_root: str = "dataset", scenario_type: str = "cybertown"):
        """Initialize task generator

        Args:
            dataset_root: Dataset root directory
            scenario_type: Scenario type
        """
        self.dataset_root = Path(dataset_root)
        self.scenario_type = scenario_type

        # Set directory paths
        self.scenarios_dir = self.dataset_root / "scenarios" / scenario_type
        self.goals_dir = self.dataset_root / "goals" / scenario_type
        self.tasks_dir = self.dataset_root / "tasks" / scenario_type

        # Ensure task directory exists
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Task generator initialized: {scenario_type}")

    def load_goals(self, goals_file: Optional[str] = None) -> List[Dict[str, Any]]:
        path = Path(goals_file) if goals_file else (self.goals_dir / "goals.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"Goals file not found: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)  # Try as JSON
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f if line.strip()]  # As JSONL

        # Deserialize goal_details
        for item in data:
            if "goal_details" in item and isinstance(item["goal_details"], str):
                try:
                    item["goal_details"] = json.loads(item["goal_details"])
                except json.JSONDecodeError:
                    pass

        return data

    def get_available_scenarios(self) -> List[str]:
        """Get available scenario list

        Returns:
            Scenario ID list
        """
        if not self.scenarios_dir.exists():
            logger.warning(f"Scenarios directory not found: {self.scenarios_dir}")
            return []

        scenarios = []
        for scenario_dir in self.scenarios_dir.iterdir():
            if scenario_dir.is_dir() and (scenario_dir / "scene_graph.json").exists():
                scenarios.append(scenario_dir.name)

        scenarios.sort()  # Sort to ensure consistent order
        logger.info(f"Found {len(scenarios)} scenarios")
        return scenarios

    def _guess_description(self, goal: Dict[str, Any]) -> str:
        """Extract description uniformly: instruction > description > ''."""
        return goal.get("instruction") or goal.get("description") or ""

    def _extract_success_condition(
            self, goal: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract success_condition uniformly: prefer root, then goal_details.success_condition."""
        if "success_condition" in goal and isinstance(goal["success_condition"], dict):
            return goal["success_condition"]
        td = goal.get("goal_details")
        if isinstance(td, dict) and isinstance(td.get("success_condition"), dict):
            return td["success_condition"]
        return None

    def _extract_context_target(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context uniformly
        """
        details = goal.get("goal_details", {}) or {}
        core = details.get("core_params")
        return deepcopy(core or {})

    def clean_goal_for_task(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        - Keep id / goal_type / goal_determinacy / description / quantifier / success_condition / context
        """
        details = goal.get("goal_details", {}) or {}
        goal_id = goal.get("id") or details.get("goal_id")
        goal_type = details.get("goal_type")
        goal_determinacy = details.get("goal_determinacy", "open")
        description = self._guess_description(goal)
        quantifier = (goal.get("goal_features") or {}).get("quantifier")
        success_condition = self._extract_success_condition(goal)
        context = self._extract_context_target(goal)

        return {
            "id": goal_id,
            "goal_type": goal_type,
            "goal_determinacy": goal_determinacy,
            "description": description,
            "quantifier": quantifier,
            "success_condition": success_condition,
            "context": context,
        }

    def generate_task(self, scenario_id: str, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single task configuration

        Args:
            scenario_id: Scenario ID
            goal: Goal configuration

        Returns:
            Task configuration
        """
        # Build task
        task = {"type": self.scenario_type, "scenario": scenario_id, "goal": goal}

        return task

    def save_task(self, task: Dict[str, Any], task_id: str) -> str:
        """
        Append records to JSONL, no longer generating individual JSON files.
        Content only keeps task_id, scenario, and goal_id.
        """
        jsonl_path = self.tasks_dir / "tasks.jsonl"
        scenario = task["scenario"]
        goal_id = task["goal"]["goal_details"]["goal_id"]

        record = {"task_id": task_id, "scenario": scenario, "goal": goal_id}

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return str(jsonl_path)

    def build_index(self):
        """Build and save task index file"""
        jsonl_path = self.tasks_dir / "tasks.jsonl"
        index_path = self.tasks_dir / "tasks.jsonl.index"

        if not jsonl_path.exists():
            return

        logger.info(f"Building index file: {index_path}")
        index = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
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
                        index[task_id] = offset
                except json.JSONDecodeError:
                    continue

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        logger.info(f"Index built, {len(index)} records total")

    def build_index(self):
        """Build and save task index file"""
        jsonl_path = self.tasks_dir / "tasks.jsonl"
        index_path = self.tasks_dir / "tasks.jsonl.index"

        if not jsonl_path.exists():
            return

        logger.info(f"Building index file: {index_path}")
        index = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
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
                        index[task_id] = offset
                except json.JSONDecodeError:
                    continue

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f)
        logger.info(f"Index built, {len(index)} records total")

    def generate_tasks_batch(
            self,
            scenarios: Optional[List[str]] = None,
            goals: Optional[List[Dict]] = None,
            goals_file: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Batch generate tasks

        Args:
            scenarios: Scenario ID list, None means use all scenarios
            goals: Goal list, None means load from file
            goals_file: Goals file path

        Returns:
            Generation results, keys are scenario IDs, values are task file path lists
        """
        # Load goals
        if goals is None:
            goals = self.load_goals(goals_file)

        # Get scenarios
        if scenarios is None:
            scenarios = self.get_available_scenarios()

        if not scenarios:
            logger.error("No available scenarios")
            return {}

        results = {}
        task_counter = 0

        # Generate tasks for each scenario-goal combination
        for scenario_id in scenarios:
            scenario_tasks = []

            for goal in goals:
                try:
                    # Generate task
                    task = self.generate_task(scenario_id, goal)

                    # Generate taskID
                    goal_id = goal.get("goal_details").get("goal_id") or f"g_{task_counter}"
                    task_id = f"{self.scenario_type}_{scenario_id}_{goal_id}"

                    # Save task
                    filepath = self.save_task(task, task_id)
                    scenario_tasks.append(filepath)
                    task_counter += 1

                except Exception as e:
                    logger.error(
                        f"Task generation failed - Scenario: {scenario_id}, Goal: {goal_id}: {e}"
                    )
                    continue

            results[scenario_id] = scenario_tasks
            logger.info(f"Scenario {scenario_id}: Generated {len(scenario_tasks)} tasks")

        total = sum(len(tasks) for tasks in results.values())
        logger.info(f"Total tasks generated: {total}")

        # Build index
        self.build_index()

        return results

    def generate_single_scenario_tasks(
            self, scenario_id: str, goals_file: Optional[str] = None
    ) -> List[str]:
        """Generate all tasks for a single scenario

        Args:
            scenario_id: Scenario ID
            goals_file: Goals file path

        Returns:
            List of generated task file paths
        """
        results = self.generate_tasks_batch(
            scenarios=[scenario_id], goals_file=goals_file
        )
        return results.get(scenario_id, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics

        Returns:
            Statistics
        """
        if not self.tasks_dir.exists():
            return {"total": 0, "by_scenario": {}, "by_goal": {}}

        task_files = list(self.tasks_dir.glob("task_*.json"))
        by_scenario = {}
        by_goal = {}

        for task_file in task_files:
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task = json.load(f)

                scenario = task.get("scenario", "unknown")
                goal_id = task.get("goal", {}).get("goal_details", {}).get("goal_id", "unknown")

                by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
                by_goal[goal_id] = by_goal.get(goal_id, 0) + 1

            except Exception as e:
                logger.warning(f"Failed to read task file {task_file}: {e}")

        return {
            "total": len(task_files),
            "by_scenario": by_scenario,
            "by_goal": by_goal,
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate multi-robot task files")

    parser.add_argument(
        "--dataset", type=str, default=os.path.join(os.path.dirname(__file__), "../../dataset/semantic"), help="Dataset root directory"
    )
    parser.add_argument("--type", type=str, default="cybertown", help="Scenario type")
    parser.add_argument(
        "--scenarios", type=str, nargs="*", help="Specify scenario ID list; if not specified, all scenarios are used"
    )
    parser.add_argument("--goals", type=str, help="Goals file path")
    parser.add_argument("--single", type=str, default="scenario_1", help="Generate tasks for a single specified scenario only")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--clean", action="store_true", help="Clean existing tasks before generation")

    args = parser.parse_args()

    # Create generator
    generator = TaskGenerator(dataset_root=args.dataset, scenario_type=args.type)

    # Clean existing tasks
    if args.clean and generator.tasks_dir.exists():
        for task_file in generator.tasks_dir.glob("task_*.json"):
            task_file.unlink()
        logger.info("Old task files cleaned")

    # Show statistics
    if args.stats:
        stats = generator.get_statistics()
        print("\n=== Task Statistics ===")
        print(f"Total tasks: {stats['total']}")

        if stats["by_scenario"]:
            print("\nBy scenario:")
            for scenario, count in sorted(stats["by_scenario"].items()):
                print(f"  {scenario}: {count}")

        if stats["by_goal"]:
            print("\nBy goal:")
            for goal_id, count in sorted(stats["by_goal"].items()):
                print(f"  {goal_id}: {count}")
        return

    # Generate task
    try:
        if args.single:
            # Single scenario mode
            tasks = generator.generate_single_scenario_tasks(
                scenario_id=args.single, goals_file=args.goals
            )
            print(f"\n✅ Generated {len(tasks)} tasks for scenario {args.single}")

        else:
            # Batch mode
            results = generator.generate_tasks_batch(
                scenarios=args.scenarios, goals_file=args.goals
            )

            total = sum(len(tasks) for tasks in results.values())
            print(f"\n✅ Task generation complete!")
            print(f"Scenarios: {len(results)}")
            print(f"Total tasks: {total}")

            # Show task count per scenario
            for scenario_id, tasks in sorted(results.items()):
                print(f"  {scenario_id}: {len(tasks)} tasks")

    except Exception as e:
        logger.error(f"Task generation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
