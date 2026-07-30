#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Generator
- Outputs segmented structured data for deduplication
- Uses dataset info without environment initialization
"""
import logging
import sys
from pathlib import Path
import re
from typing import Optional, Set, Dict, Any, List

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.task_solver.sgi_planner.prompt.runtime_builders import (
    compose_master_context,
    select_prompt_and_feedback,
    select_prompt_components,
    PromptComponents,
)
from modules.task_solver.sgi_planner.utils import to_concise_robot_info
from modules.task_solver.sgi_planner.prompt.observation import format_observation
from modules.task_solver.world_model.world_model_manager import WorldModelManager
from modules.platform.semantic_platform.scene_graph_manager import SemanticSceneGraph


class DatasetPromptGenerator:
    """
    Dataset Prompt Generator
    - Outputs segmented structured data directly
    - Uses dataset info without environment initialization
    - Supports segment-level deduplication
    """

    def __init__(
        self,
        planner_mode: str = "phase",
        use_environment_model: bool = True,
        robot_type_list: Optional[List[str]] = None,
    ):
        """
        Initialize Prompt Generator

        Args:
            planner_mode: Planning mode ("phase" | "full")
            use_environment_model: Whether to use environment model
            robot_type_list: Robot type list for filtering robot_labels
        """
        self.planner_mode = planner_mode
        self.use_environment_model = use_environment_model
        self.robot_type_list = robot_type_list or ["UAV", "UGV", "Quadruped", "Humanoid"]

        # World model cache: scenario_id -> WorldModelManager (reuse environment info)
        self._world_model_cache: Dict[str, WorldModelManager] = {}

    def _get_available_robots(
        self,
        scenario_data: Dict[str, Any],
        types: Optional[List[str]] = None,
        concise: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        # Normalize robot types (case, aliases, etc.)
        def canonicalize(t: Optional[str]) -> Optional[str]:
            if not t:
                return None
            t_norm = t.strip().lower().replace(" ", "").replace("-", "_")
            aliases = {
                "uav": "UAV",
                "fw_uav": "FW_UAV",
                "fwuav": "FW_UAV",
                "fixedwing": "FW_UAV",
                "fixed_wing": "FW_UAV",
                "ugv": "UGV",
                "quadruped": "Quadruped",
                "quadrupte": "Quadruped",
                "quad": "Quadruped",
                "humanoid": "Humanoid",
            }
            return aliases.get(t_norm)

        # Extract numeric suffix from labels (e.g., UAV_03 -> 3)
        id_tail_re = re.compile(r"(\d+)$")

        def extract_numeric_tail(label: str) -> Optional[int]:
            m = id_tail_re.search(label.strip())
            return int(m.group(1)) if m else None

        # Type filtering
        include_set: Optional[Set[str]] = None
        if types is not None:
            include_set = {c for c in (canonicalize(t) for t in types) if c is not None}

        # Read all nodes
        nodes = (
            scenario_data.get("nodes")
            or scenario_data.get("graph", {}).get("nodes")
            or scenario_data.get("scene_nodes")
            or []
        )

        detailed: Dict[str, Dict[str, Any]] = {}

        # Iterate nodes and filter available robots
        for node in nodes:
            props = node.get("properties", {})
            if props.get("category") != "robot":
                continue

            # Status check: must be healthy, battery>=20, comm not jammed
            if (
                props.get("status") == "error"
                or props.get("battery_level") < 20.0
                or props.get("comm") == "jammed"
            ):
                continue

            robot_type_raw = props.get("type")
            robot_label = props.get("label") or node.get("id")
            canonical_type = canonicalize(robot_type_raw)
            if not canonical_type or not robot_label:
                continue

            # Type filtering
            if include_set is not None and canonical_type not in include_set:
                continue

            # Record robot
            if canonical_type not in detailed:
                detailed[canonical_type] = {"labels": [], "num": 0}

            detailed[canonical_type]["labels"].append(robot_label)
            detailed[canonical_type]["num"] += 1

        # Full mode: return directly
        if not concise:
            return detailed

        # Concise mode: return numeric list
        concise_map: Dict[str, List[int]] = {}
        for rtype, info in detailed.items():
            nums = []
            for label in info["labels"]:
                n = extract_numeric_tail(label)
                if n is not None:
                    nums.append(n)
            if nums:
                concise_map[rtype] = nums

        return concise_map

    def _get_or_init_world_model(
        self,
        scenario_id: str,
        scenario_data: Dict[str, Any],
    ) -> WorldModelManager:
        """
        Get or initialize WorldModelManager for a scenario (cached, reuses environment info).

        Args:
            scenario_id: Scenario ID
            scenario_data: Raw scenario data (contains nodes and edges)

        Returns:
            Initialized WorldModelManager instance
        """
        if scenario_id in self._world_model_cache:
            return self._world_model_cache[scenario_id]

        # Build SemanticSceneGraph from scenario data
        nodes = scenario_data.get("nodes", [])
        edges = scenario_data.get("edges", [])
        logger = logging.getLogger(f"dataset_builder.{scenario_id}")

        # SemanticSceneGraph is singleton, reset before creating new instance
        SemanticSceneGraph._instance = None
        scene_graph = SemanticSceneGraph(
            initial_nodes=nodes,
            initial_edges=edges,
        )

        # Initialize WorldModelManager (local knowledge mode)
        wmm = WorldModelManager(scene_graph=scene_graph, logger=logger)
        wmm.initialize_local_knowledge()

        self._world_model_cache[scenario_id] = wmm
        return wmm

    def _get_scene_description(
        self,
        scenario_id: str,
        scenario_data: Dict[str, Any],
    ) -> str:
        """
        Generate environment description using format_observation (consistent with runtime).

        Args:
            scenario_id: Scenario ID
            scenario_data: Raw scenario data

        Returns:
            Formatted environment observation string
        """
        wmm = self._get_or_init_world_model(scenario_id, scenario_data)
        known_nodes = wmm.known_nodes
        known_edges = wmm.known_edges

        # Compute robot_labels (consistent with solver_context.py)
        robot_nodes = [
            n for n in known_nodes
            if (n.get('properties') or {}).get('category') == 'robot'
        ]
        robot_labels = sorted(
            n.get('properties', {}).get('label', '')
            for n in robot_nodes
            if n.get('properties', {}).get('label')
            and n.get('properties', {}).get('type') in self.robot_type_list
        )

        return format_observation(known_nodes, known_edges, robot_labels)

    def _generate_master_context_segments(
        self,
        scene_desc: Optional[str] = None,
        goal_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate fine-grained segments of master_context (for better deduplication)

        Splits master_context into:
        1. skill_set_markdown: Same for all tasks
        2. env_description: Same per scenario
        3. goal_type_notes: Grouped by goal_type (only a few types)
        4. core_definitions: Grouped by goal_type and planner_mode
        5. universal_output_rules: May contain common output rules (if any)

        Args:
            scene_desc: Scene description
            goal_type: Goal type

        Returns:
            Dictionary containing all segments
        """
        from modules.task_solver.sgi_planner.utils import skill_library_to_markdown
        from modules.task_solver.sgi_planner.prompt.runtime_builders import (
            robot_skill_library,
            _select_goal_notes,
        )
        from modules.task_solver.sgi_planner.prompt import (
            build_core_definitions,
            build_core_definitions_replanning,
        )

        # 1. skill_set_markdown (same for all tasks)
        skill_set_markdown = skill_library_to_markdown(
            robot_skill_library, include_details=False,
        )

        # 2. env_description (same per scenario)
        env_description = scene_desc or "No scene description available"

        # 3. goal_type_notes (grouped by goal_type)
        goal_type_name, goal_type_notes = _select_goal_notes(goal_type)

        # 4. core_definitions (grouped by planner_mode)
        core_definitions = build_core_definitions(goal_type_notes)
        universal_output_rules = ""

        # 5. Generate full master_context
        full_master_context = compose_master_context(
            planner_mode=self.planner_mode,
            use_environment_model=self.use_environment_model,
            scene_desc=scene_desc,
            goal_type=goal_type,
        )

        return {
            "skill_set_markdown": skill_set_markdown,
            "env_description": env_description,
            "goal_type_notes": goal_type_notes,
            "core_definitions": core_definitions,
            "universal_output_rules": universal_output_rules,
            "full_master_context": full_master_context,
        }

    def generate_prompt_segments(
        self,
        task_data: Dict[str, Any],
        scenario_data: Dict[str, Any],
        type_name: str,
        scenario_id: str,
        goal_id: str,
    ) -> Dict[str, Any]:
        """
        Generate segmented structured prompt data (for deduplication)

        Args:
            task_data: Task data
            scenario_data: Scenario data
            type_name: Scenario type
            scenario_id: Scenario ID
            goal_id: Goal ID

        Returns:
            Dictionary containing segmented data and metadata
        """
        # 1. Extract goal info
        goal = task_data.get("goal", {})
        instruction = goal.get("instruction", "")
        goal_type = goal.get("goal_details", {}).get("goal_type")

        # 2. Generate scene description (using format_observation, consistent with runtime)
        scene_desc = None
        if self.use_environment_model:
            scene_desc = self._get_scene_description(scenario_id, scenario_data)

        # 3. Generate fine-grained segments of master_context (for better deduplication)
        master_segments = self._generate_master_context_segments(
            scene_desc=scene_desc,
            goal_type=goal_type,
        )

        # 4. Get available robots (extracted from scenario data, segment 2)
        available_robots_dict = self._get_available_robots(scenario_data, types=self.robot_type_list)
        available_robots_text = to_concise_robot_info(available_robots_dict)

        # 5. Get Prompt components (Head, Format) - explicit interface
        prompt_components: PromptComponents = select_prompt_components(
            planner_mode=self.planner_mode,
            use_separate_prompts=True, # Separate templates for initial planning and replanning
            is_replanning=False, # Initial planning
        )

        # 6. feedback_context (segment 3, empty for initial planning)
        feedback_context_section = ""

        # 7. instruction (segment 4)
        # instruction is already separate text

        # Return segmented data (includes fine-grained master_context segments and response_format)
        return {
            "segments": {
                # Fine-grained master_context segments (for deduplication)
                # These segments will be deduplicated as they're in deduplicated_fields
                "master_skill_set": master_segments["skill_set_markdown"],
                "master_env_description": master_segments["env_description"],
                "master_goal_type_notes": master_segments["goal_type_notes"],
                "master_core_definitions": master_segments["core_definitions"],
                "master_universal_rules": master_segments.get(
                    "universal_output_rules", ""
                ),
                # Response Format component (for deduplication)
                "response_format": prompt_components.format,
                # Other segments (stored directly, not deduplicated)
                "available_robots": available_robots_text,
                "feedback_context": feedback_context_section,
                "instruction": instruction,
                # Head template (lightweight, stored directly)
                "prompt_head_template": prompt_components.head,
                # Note: master_context contains all segments to be deduplicated
                # In dedup mode, it shouldn't be stored directly (not in direct_fields)
                # Can be reconstructed from fine-grained segments when needed (using _reconstruct_master_context_from_segments)
                # Kept here only for backward compatibility with non-dedup mode
                "master_context": master_segments["full_master_context"],
            },
            "template_info": {
                "planner_mode": self.planner_mode,
                "head_template": prompt_components.head,  # Head template (lightweight)
                "response_format": prompt_components.format,  # Response format (deduplicated)
                # Keep full template for backward compatibility
                "template": prompt_components.head + "\n" + prompt_components.format,
            },
            "metadata": {
                "goal_id": goal_id,
                "task_id": f"{type_name}_{scenario_id}_{goal_id}",
                "scenario_type": type_name,
                "scenario_id": scenario_id,
                "goal_type": goal_type,
                "instruction": instruction,
            },
        }

    def build_full_prompt(self, segments_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build full prompt from segmented data (on-the-fly assembly)

        Uses Head.format(...) + Format logic to assemble full prompt in memory.
        Format is retrieved from dedup pool (if using dedup mode).

        Args:
            segments_data: Segmented data

        Returns:
            Full prompt data
        """
        segments = segments_data["segments"]
        template_info = segments_data["template_info"]

        # Prefer full master_context if exists, otherwise reconstruct from fine-grained segments
        if "master_context" in segments and segments["master_context"]:
            master_context = segments["master_context"]
        else:
            # Reconstruct master_context from fine-grained segments
            master_context = self._reconstruct_master_context_from_segments(segments)

        # Get Head template (lightweight, stored directly) - reject any manual splitting
        head_template = template_info.get("head_template") or segments.get(
            "prompt_head_template", ""
        )

        # Get Format (from dedup pool or stored directly) - reject any manual splitting
        response_format = template_info.get("response_format") or segments.get(
            "response_format", ""
        )

        # Validate required components exist (reject manual splitting, error directly)
        if not head_template:
            raise ValueError(
                "Missing head_template: Cannot build full prompt. "
                "Please ensure select_prompt_components is used to obtain predefined Head components."
            )
        if not response_format:
            raise ValueError(
                "Missing response_format: Cannot build full prompt. "
                "Please ensure select_prompt_components is used to obtain predefined Format components."
            )

        # On-the-fly assembly: Head.format(...) + Format (using predefined components only), initial planning
        formatted_head = head_template.format(
            master_context=master_context,
            available_robots=segments["available_robots"],
            feedback_context_section=segments["feedback_context"],
            instruction=segments["instruction"],
        )
        formatted_response = response_format.format()
        # Assemble full prompt
        full_prompt = f"{formatted_head}\n\n{formatted_response}"

        return [{"role": "user", "content": full_prompt.strip()}]

    def _reconstruct_master_context_from_segments(
        self, segments: Dict[str, str]
    ) -> str:
        """
        Reconstruct full master_context from fine-grained segments

        Args:
            segments: Segmented data

        Returns:
            Full master_context string
        """
        from modules.task_solver.sgi_planner.prompt import (
            master_text,
            master_text_no_env,
            master_text_full,
            master_text_full_no_env,
        )

        skill_set = segments.get("master_skill_set", "")
        env_desc = segments.get("master_env_description", "")
        core_def = segments.get("master_core_definitions", "")
        universal_rules = segments.get("master_universal_rules", "")

        if self.planner_mode == "phase":
            if self.use_environment_model:
                return master_text.format(
                    env_description=env_desc,
                    skill_set_markdown=skill_set,
                    core_definitions=core_def,
                )
            else:
                return master_text_no_env.format(
                    skill_set_markdown=skill_set,
                    core_definitions=core_def,
                )
        else:
            # full mode
            if self.use_environment_model:
                return master_text_full.format(
                    env_description=env_desc,
                    skill_set_markdown=skill_set,
                    core_definitions=core_def,
                    graph_conventions_full="",  # 需要从segments获取
                )
            else:
                return master_text_full_no_env.format(
                    skill_set_markdown=skill_set,
                    core_definitions=core_def,
                    graph_conventions_full="",  # 需要从segments获取
                )
