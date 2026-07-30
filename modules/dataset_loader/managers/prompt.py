import json
from typing import Any, Dict, List, Optional
from .base import BaseDataManager

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


class PromptManager(BaseDataManager):
    """
    Prompt Manager (Pure Compression Mode)

    Assumptions:
    1. Input data is fully compressed format (contains only indices, no segments dict).
    2. Must dynamically assemble master_context at runtime.
    3. get_prompt returns decompressed data and formatted string in one call.
    """

    def __init__(self, dataset: Dataset, pools: Dict[str, List[str]]):
        super().__init__(dataset, key_column="task_id")
        self.pools = pools

        # 映射配置: Pool Name -> Segment Key
        self.pool_map = {
            "skill_set": "skill_set",
            "env_desc": "env_desc",
            "goal_notes": "goal_notes",
            "core_def": "core_def",
            "univ_rules": "univ_rules",
            "available_robots": "available_robots",
            "response_format": "response_format",
            "head_template": "head_template",
        }

        # 直接字段: 不需要查表，直接从 record 复制
        self.direct_fields = ["instruction", "feedback_context"]

    def get_prompt(self, task_id: str) -> Dict[str, Any]:
        """
        Get task Prompt, perform decompression, assembly and formatting

        Returns:
            {
                "task_id": str,
                "metadata": dict,
                "segments": dict,  # Contains all text segments
                "prompt": str      # Final formatted complete string
            }
        """
        record = self.get_by_id(task_id)
        if not record:
            raise ValueError(f"Task {task_id} not found")

        return self._inflate_and_format(record)

    def _inflate_and_format(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Core processing pipeline: decompress -> assemble context -> format string"""

        # 1. Initialize result container
        segments = {}
        metadata = record.get("metadata", {})

        # 2. Decompress Pool fields (Index -> Text)
        for pool_name, segment_key in self.pool_map.items():
            idx_key = f"{pool_name}_idx"
            idx = record.get(idx_key)

            # Strict mode: if index exists, must be valid
            if idx is not None:
                try:
                    segments[segment_key] = self.pools[pool_name][idx]
                except IndexError:
                    raise ValueError(
                        f"Invalid index {idx} for pool {pool_name} in task {record['task_id']}"
                    )
            else:
                # If index missing, decide based on business logic whether to error or fill empty
                # For robustness, provide default empty value, except templates must error
                if segment_key in ["head_template", "response_format"]:
                    raise ValueError(f"Missing required index field: {idx_key}")
                segments[segment_key] = ""

        # =========================================================
        # Step 2.5: Update core_def
        # Inject goal_notes into core_def template
        # =========================================================
        goal_notes_text = segments.get("goal_notes", "")

        # 3. Copy direct fields
        for field in self.direct_fields:
            segments[field] = record.get(field, "")

        # 4. Dynamically assemble Master Context (Runtime Logic)
        # Must be done after decompression, as it needs decompressed env_desc
        segments["master_context"] = self._build_master_context(segments, metadata, goal_notes_text=goal_notes_text)
        # 5. Final formatting (String Interpolation)
        full_prompt_string = self._format_final_string(segments)

        return {
            "task_id": record["task_id"],
            "type": record.get("type", "cybertown"),
            "metadata": metadata,
            "segments": segments,
            "prompt": full_prompt_string,  # Return complete Prompt directly
        }

    @staticmethod
    def _build_master_context(
            segments: Dict[str, str], metadata: Dict[str, Any],goal_notes_text: str
    ) -> str:
        """Call business logic to assemble master_context"""
        try:
            from modules.task_solver.sgi_planner.prompt.runtime_builders import (
                compose_master_context,
            )

            scene_desc = segments.get("env_desc")
            if not scene_desc or scene_desc == "No scene description available":
                scene_desc = None

            # Note: In compression mode, template_info is usually not stored in record
            # Use default config here, or should inject Config from outside.
            # Assume full default config was used during generation.
            return compose_master_context(
                planner_mode="full",  # 同上
                use_environment_model=True,
                scene_desc=scene_desc,
                goal_notes_text=goal_notes_text,
                goal_type=metadata.get("goal_type"),
                is_replanning=False,
            )
        except ImportError:
            # Dev environment returns placeholder if module missing
            return (
                f"[Master Context Placeholder: {segments.get('env_desc', 'No Desc')}]"
            )
        except Exception as e:
            return f"[Error building master context: {e}]"

    @staticmethod
    def _format_final_string(segments: Dict[str, str]) -> str:
        """Execute final string replacement"""
        head_template = segments.get("head_template", "")
        response_format = segments.get("response_format", "")

        if not head_template or not response_format:
            return "Error: Missing template components"

        try:
            # Use segments dict unpacking to fill template
            # Placeholder names in template must match segments keys
            # (master_context, available_robots, feedback_context_section, instruction)

            # Map some aliases for template naming differences
            format_kwargs = {
                "master_context": segments.get("master_context", ""),
                "available_robots": segments.get("available_robots", ""),
                "feedback_context_section": segments.get(
                    "feedback_context", ""
                ),  # Template uses this name
                "instruction": segments.get("instruction", ""),
            }

            formatted_head = head_template.format(**format_kwargs)
            response_format = response_format.format()
            return f"{formatted_head}\n\n{response_format}".strip()

        except KeyError as e:
            return f"Format Error: Template expects missing key {e}"
