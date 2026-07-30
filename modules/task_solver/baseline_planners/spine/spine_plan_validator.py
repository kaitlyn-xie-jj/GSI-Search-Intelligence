# -*- coding: utf-8 -*-
"""
SPINE Plan Validator

Validates each skill_str against regex patterns defined in SKILL_SCHEMAS,
and checks that robot labels are in the available list.
"""
import logging
from typing import Dict, List, Tuple

from modules.task_solver.sgi_planner.prompt.atomic_skills import SKILL_SCHEMAS

logger = logging.getLogger(__name__)


class SPINEPlanValidator:
    """Validate SPINE-generated timestep plans against skill format.

    Responsibilities:
    1. Check each "robot:skill" entry's skill_str matches a pattern in SKILL_SCHEMAS
    2. Check robot labels are in the available list
    3. Return validation result and error feedback
    """

    def validate(
        self,
        plan: List[List[str]],
        robot_labels: List[str],
    ) -> Tuple[bool, str]:
        """Validate a timestep plan.

        Args:
            plan: [["UAV-1:take_off", "UGV-1:navigate<loc>"], ["UAV-1:search<a>_for<t>"], ...]
            robot_labels: Available robot label list

        Returns:
            (is_valid, error_feedback): is_valid is True if validation passes,
            error_feedback is empty string; otherwise contains descriptive error info.
        """
        errors: List[str] = []

        for ts_idx, timestep in enumerate(plan):
            if not isinstance(timestep, list):
                errors.append(
                    f"Timestep {ts_idx}: must be a list, got {type(timestep).__name__}"
                )
                continue

            for entry in timestep:
                if not isinstance(entry, str):
                    errors.append(
                        f"Timestep {ts_idx}: entry must be a string, got {type(entry).__name__}"
                    )
                    continue

                if ":" not in entry:
                    errors.append(
                        f"Timestep {ts_idx}: entry '{entry}' missing ':' separator. "
                        f"Expected format: 'robot_label:skill'"
                    )
                    continue

                label, skill_str = entry.split(":", 1)
                label = label.strip()
                skill_str = skill_str.strip()

                # Validate robot label
                if label not in robot_labels:
                    errors.append(
                        f"Timestep {ts_idx}: invalid robot label '{label}'. "
                        f"Available robots: {robot_labels}"
                    )

                # Validate skill format
                if not self._match_skill(skill_str):
                    expected = self._format_expected_patterns()
                    errors.append(
                        f"Timestep {ts_idx}, robot '{label}': invalid skill '{skill_str}'. "
                        f"Expected formats: {expected}"
                    )

        if errors:
            error_feedback = "Plan validation failed:\n" + "\n".join(
                f"  - {e}" for e in errors
            )
            logger.warning(error_feedback)
            return False, error_feedback

        return True, ""

    @staticmethod
    def _match_skill(skill_str: str) -> bool:
        """Check if skill_str matches any pattern in SKILL_SCHEMAS."""
        return any(
            schema["pattern"].match(skill_str)
            for schema in SKILL_SCHEMAS.values()
        )

    @staticmethod
    def _format_expected_patterns() -> str:
        """Generate a brief description of available skill formats from SKILL_SCHEMAS regex patterns."""
        examples = []
        for name, schema in SKILL_SCHEMAS.items():
            # Extract readable format from regex pattern
            pattern_str = schema["pattern"].pattern
            # Strip end-of-line anchor $
            readable = pattern_str.rstrip("$")
            # Replace capture groups ([^>]+) with param names
            for param in schema["params"]:
                readable = readable.replace("([^>]+)", param, 1)
            examples.append(readable)
        return ", ".join(examples)
