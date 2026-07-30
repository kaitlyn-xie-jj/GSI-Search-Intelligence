from typing import Optional, Tuple, NamedTuple

# Import master templates and convention sections.
from . import (
    # phase
    master_text,
    master_text_no_env,
    # full
    master_text_full,
    master_text_full_no_env,
    graph_conventions_full,
    # Assembly functions
    build_core_definitions,
    build_core_definitions_replanning,
    # Prompt templates
    # One shared template for initial planning and replanning
    PHASE_TASK_PLAN_TEMPLATE,
    FULL_TASK_PLAN_TEMPLATE,
    # Separate initial-planning templates
    INITIAL_PHASE_TASK_PLAN_TEMPLATE,
    INITIAL_FULL_TASK_PLAN_TEMPLATE,
    # Separate replanning templates
    REPLANNING_PHASE_TASK_PLAN_TEMPLATE,
    REPLANNING_FULL_TASK_PLAN_TEMPLATE,
    # Feedback template
    feedback_plan_context,
    # Skill library
    robot_skill_library,
)
from modules.task_solver.sgi_planner.utils import skill_library_to_markdown
from .goal_type_notes import GOAL_TYPE_NOTES


class PromptComponents(NamedTuple):
    """Prompt components: Head template and response format."""

    head: str  # Head template containing master_context, available_robots, instruction, etc.
    format: str  # Response format section, starting from "## Core Directives" or "## Persona and Goal".


def _select_goal_notes(goal_type: Optional[str]) -> tuple[str, str]:
    if not goal_type:
        return ("default", GOAL_TYPE_NOTES.get("__default__", ""))

    key = str(goal_type).strip().lower()
    if key in GOAL_TYPE_NOTES:
        return (key, GOAL_TYPE_NOTES[key])

    # Simple category fallback.
    if (
        "parking" in key
        or "violation" in key
        or "crowd" in key
        or key.startswith("event.")
    ):
        return ("event", GOAL_TYPE_NOTES["event"])
    return (
        "object",
        GOAL_TYPE_NOTES.get("object", GOAL_TYPE_NOTES.get("__default__", "")),
    )


def compose_master_context(
    *,
    planner_mode: str,
    use_environment_model: bool,
    scene_desc: Optional[str] = None,
    goal_type: Optional[str] = None,
    goal_notes_text: Optional[str] = None,
    is_replanning: bool = False,
) -> str:
    """Return the formatted master_context string."""
    skill_set_markdown = skill_library_to_markdown(
        robot_skill_library, include_details=False,
    )
    env_text = scene_desc or "No scene description available"
    if goal_notes_text is not None:
        goal_type_notes=goal_notes_text
    else:
        goal_type_name, goal_type_notes = _select_goal_notes(goal_type)

    if is_replanning:
        # Replanning.
        filled_core = build_core_definitions_replanning(goal_type_notes)
        if planner_mode == "phase":
            if use_environment_model:
                return master_text.format(
                    env_description=env_text,
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                )
            else:
                return master_text_no_env.format(
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                )
        elif planner_mode == "full":
            if use_environment_model:
                return master_text_full.format(
                    env_description=env_text,
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                    graph_conventions_full=graph_conventions_full,
                )
            else:
                return master_text_full_no_env.format(
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                    graph_conventions_full=graph_conventions_full,
                )
    else:
        # Initial planning.
        filled_core = build_core_definitions(goal_type_notes)
        if planner_mode == "phase":
            if use_environment_model:
                return master_text.format(
                    env_description=env_text,
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                )
            else:
                return master_text_no_env.format(
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                )

        if planner_mode == "full":
            if use_environment_model:
                return master_text_full.format(
                    env_description=env_text,
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                    graph_conventions_full=graph_conventions_full,
                )
            else:
                return master_text_full_no_env.format(
                    skill_set_markdown=skill_set_markdown,
                    core_definitions=filled_core,
                    graph_conventions_full=graph_conventions_full,
                )

    raise ValueError(f"Unknown planner_mode: {planner_mode}")


def select_prompt_and_feedback(
    *,
    planner_mode: str,
    use_separate_prompts: bool = False,
    is_replanning: bool = False,
) -> Tuple[str, str]:
    """
    Select only the prompt template and feedback snippet; do not build master.
    - use_separate_prompts=False: initial planning and replanning share one template.
    - use_separate_prompts=True: initial planning uses INITIAL_*, replanning uses REPLANNING_*.
    """
    if not use_separate_prompts:
        if planner_mode == "phase":
            return PHASE_TASK_PLAN_TEMPLATE, feedback_plan_context
        if planner_mode == "full":
            return FULL_TASK_PLAN_TEMPLATE, feedback_plan_context
        raise ValueError(f"Unknown planner_mode: {planner_mode}")
    else:
        if planner_mode == "phase":
            if is_replanning:
                return REPLANNING_PHASE_TASK_PLAN_TEMPLATE, feedback_plan_context
            else:
                return INITIAL_PHASE_TASK_PLAN_TEMPLATE, feedback_plan_context
        if planner_mode == "full":
            if is_replanning:
                return REPLANNING_FULL_TASK_PLAN_TEMPLATE, feedback_plan_context
            else:
                return INITIAL_FULL_TASK_PLAN_TEMPLATE, feedback_plan_context

    raise ValueError(f"Unknown planner_mode: {planner_mode}")


def select_prompt_components(
    *,
    planner_mode: str,
    use_separate_prompts: bool = False,
    is_replanning: bool = False,
) -> PromptComponents:
    """
    Explicitly return structured Prompt components (Head, Format). Only full planning mode is supported.

    Args:
        planner_mode: Planning mode. Only "full" is supported.
        use_separate_prompts: Whether to use separate initial/replanning templates.
        is_replanning: Whether this is replanning mode.

    Returns:
        PromptComponents: Structured components containing head and format.
    """
    # Validate parameters.
    if planner_mode != "full":
        raise ValueError(f"Only planner_mode='full' is supported, current: {planner_mode}")

    # Import template components.
    from .task_plan_prompt import (
        TASK_PLAN_HEAD,
        PHASE_TASK_PLAN_RESPONSE_FORMAT,
        FULL_TASK_PLAN_RESPONSE_FORMAT,
        INITIAL_TASK_PLAN_HEAD,
        INITIAL_PHASE_TASK_PLAN_RESPONSE_FORMAT,
        INITIAL_FULL_TASK_PLAN_RESPONSE_FORMAT,
        REPLANNING_TASK_PLAN_HEAD,
        REPLANNING_PHASE_TASK_PLAN_RESPONSE_FORMAT,
        REPLANNING_FULL_TASK_PLAN_RESPONSE_FORMAT,
    )

    # Full mode: use only predefined Head and Format.
    if use_separate_prompts:
        if is_replanning:
            return PromptComponents(head=REPLANNING_TASK_PLAN_HEAD, format=REPLANNING_FULL_TASK_PLAN_RESPONSE_FORMAT)
        else:
            return PromptComponents(head=INITIAL_TASK_PLAN_HEAD, format=INITIAL_FULL_TASK_PLAN_RESPONSE_FORMAT)
    else:
        return PromptComponents(head=TASK_PLAN_HEAD, format=FULL_TASK_PLAN_RESPONSE_FORMAT)
