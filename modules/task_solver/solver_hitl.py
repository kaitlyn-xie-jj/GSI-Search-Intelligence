# -*- coding: utf-8 -*-
"""
Task solver HITL interaction module.
Handles human-in-the-loop instruction input, plan review, and result decisions.
"""
import json
from typing import Dict, Optional, Any

from modules.platform.platform_factory import get_scene_graph
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.utils.system.logging_utils import dlog
from modules.communication.enums import ReviewType, DecisionType


# ---------------------------------------------------------------------------
# Decision type → human-readable description mapping
# ---------------------------------------------------------------------------
DECISION_DESCRIPTIONS: Dict[DecisionType, str] = {
    DecisionType.SEARCH_NOT_FOUND: "Skill list execution completed, search did not find the target, end task?",
    DecisionType.SEARCH_COMPLETED: "Skill list execution completed, search completed, end task?",
    DecisionType.PHOTO_COMPLETED: "Skill list execution completed, photo taken, end task?",
    DecisionType.FOLLOW_COMPLETED: "Skill list execution completed, follow completed, end task?",
    DecisionType.TRANSPORT_COMPLETED: "Skill list execution completed, transport completed, end task?",
    DecisionType.GUIDE_COMPLETED: "Skill list execution completed, guide completed, end task?",
    DecisionType.BROADCAST_COMPLETED: "Skill list execution completed, verbal broadcast completed, end task?",
    DecisionType.PATROL_COMPLETED: "Skill list execution completed, patrol completed, end task?",
    DecisionType.ASSEMBLY_COMPLETED: "Skill list execution completed, assembly completed, end task?",
    DecisionType.ENFORCEMENT_COMPLETED: "Skill list execution completed, traffic enforcement completed, end task?",
    DecisionType.EVIDENCE_COMPLETED: "Skill list execution completed, evidence collection completed, end task?",
    DecisionType.EMERGENCY_COMPLETED: "Skill list execution completed, emergency response completed, end task?",
    DecisionType.TASK_COMPLETED: "Skill list execution completed, end task?",
}

# ---------------------------------------------------------------------------
# Goal type string → DecisionType mapping
# ---------------------------------------------------------------------------
GOAL_TYPE_TO_DECISION: Dict[str, DecisionType] = {
    "area_search": DecisionType.SEARCH_COMPLETED,
    "target_following": DecisionType.FOLLOW_COMPLETED,
    "transport": DecisionType.TRANSPORT_COMPLETED,
    "guidance": DecisionType.GUIDE_COMPLETED,
    "verbal_broadcast": DecisionType.BROADCAST_COMPLETED,
    "patrol": DecisionType.PATROL_COMPLETED,
    "assembly": DecisionType.ASSEMBLY_COMPLETED,
    "traffic_enforcement": DecisionType.ENFORCEMENT_COMPLETED,
    "evidence_collection": DecisionType.EVIDENCE_COMPLETED,
    "emergency_response": DecisionType.EMERGENCY_COMPLETED,
}


class SolverHITLMixin:
    """
    HITL interaction mixin.
    Provides human-in-the-loop instruction waiting, plan review, and search decisions.
    """

    # =========================================================================
    # Initialization: inject review callbacks into planning_layer
    # =========================================================================

    def _setup_review_hooks(self) -> None:
        """Inject review callbacks into planning_layer so normal and replay modes share review logic."""
        self.planning_layer.review_task_graph = self._review_task_graph
        self.planning_layer.review_skill_list = self._review_skill_list

    # =========================================================================
    # Plan review
    # =========================================================================

    async def _review_task_graph(self, task_plan: Any) -> Any:
        """
        Review the task graph if HITL review is enabled.

        Args:
            task_plan: Task plan to review, as a string or dictionary.

        Returns:
            Reviewed task plan, possibly modified.
        """
        if not self.hitl_manager or not self.hitl_manager.is_review_enabled:
            return task_plan

        try:
            plan_data = json.loads(task_plan) if isinstance(task_plan, str) else task_plan

            dlog("Requesting HITL review for task graph...", logger=self.logger, level="info")
            reviewed_data = await self.hitl_manager.request_review(
                ReviewType.TASK_GRAPH,
                plan_data
            )

            if reviewed_data != plan_data:
                dlog("Task graph was modified during review", logger=self.logger, level="info")
                self.context._generated_text['task_plan'] = (
                    json.dumps(reviewed_data, ensure_ascii=False)
                    if isinstance(reviewed_data, dict) else reviewed_data
                )
                return reviewed_data

            return task_plan

        except Exception as e:
            dlog(f"Error during task graph review: {e}", logger=self.logger, level="warning")
            return task_plan

    async def _review_skill_list(self, alloc_results: Any) -> Any:
        """
        Review skill allocation if HITL review is enabled.

        Args:
            alloc_results: Allocation results to review.

        Returns:
            Reviewed allocation results, possibly modified.
        """
        if not self.hitl_manager or not self.hitl_manager.is_review_enabled:
            return alloc_results

        try:
            alloc_data = json.loads(alloc_results) if isinstance(alloc_results, str) else alloc_results

            dlog("Requesting HITL review for skill allocation...", logger=self.logger, level="info")
            reviewed_data = await self.hitl_manager.request_review(
                ReviewType.SKILL_LIST,
                alloc_data
            )

            if reviewed_data != alloc_data:
                dlog("Skill allocation was modified during review", logger=self.logger, level="info")
                self.context._generated_text['alloc_results'] = (
                    json.dumps(reviewed_data, ensure_ascii=False)
                    if isinstance(reviewed_data, dict) else reviewed_data
                )
                return reviewed_data

            return alloc_results

        except Exception as e:
            dlog(f"Error during skill list review: {e}", logger=self.logger, level="warning")
            return alloc_results

    # =========================================================================
    # Replay mode: plan review after restoring context from episode data
    # =========================================================================

    async def _replay_hitl_review(self, episode: Dict) -> None:
        """
        Plan review in replay mode.

        Precondition: _load_episode_data has restored task_plan and alloc_results
        to context. This method only calls the shared review interface.
        """
        # Call the shared review interface.
        task_plan = self.context._generated_text.get("task_plan")
        if task_plan:
            await self._review_task_graph(task_plan)

        alloc_results = self.context._generated_text.get("alloc_results")
        if alloc_results:
            await self._review_skill_list(alloc_results)

    # =========================================================================
    # Instruction waiting
    # =========================================================================

    async def _wait_for_hitl_instruction(self) -> None:
        """
        Wait for HITL instruction input if enabled.

        When instruction_enabled is true, block until a user instruction arrives.
        After receiving it, update the instruction through _update_hitl_instruction.
        """
        if not self.hitl_manager or not self.hitl_manager.is_instruction_enabled:
            return

        dlog("Waiting for HITL instruction input...", logger=self.logger, level='stage')

        try:
            instruction_data = await self.hitl_manager.wait_for_instruction()

            if instruction_data:
                instruction_text = instruction_data.get("instruction_text", "")
                if instruction_text:
                    self._update_hitl_instruction(instruction_text)
            else:
                dlog("HITL instruction timeout, using pre-loaded instruction",
                     logger=self.logger, level='warning')
        except Exception as e:
            dlog(f"Error waiting for HITL instruction: {e}", logger=self.logger, level='error')

    # =========================================================================
    # Shared decisions
    # =========================================================================

    def _get_current_goal_type(self) -> Optional[str]:
        """Get the current goal_type string.

        Returns:
            goal_type string, or None if unavailable.
        """
        scene_graph: AbstractSceneGraph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        if not goal:
            return None
        # goal_type may be at the top level or nested in goal_details.
        return goal.get("goal_type")

    def _determine_decision_type(self) -> Optional[DecisionType]:
        """Determine decision type from the current goal type, prioritizing search_not_found.

        When goal_monitor detects search_not_found, return SEARCH_NOT_FOUND
        regardless of goal_type. Otherwise map goal_type to its decision type.

        Returns:
            DecisionType, or None when no decision is needed.
        """
        scene_graph: AbstractSceneGraph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        if not goal:
            return None

        goal_id = goal.get("id")
        if not goal_id:
            return None

        goal_monitor = self.world_model_layer.goal_progress_monitor

        # search_not_found has the highest priority.
        if goal_monitor and goal_monitor.has_search_not_found(goal_id):
            return DecisionType.SEARCH_NOT_FOUND

        # Map by goal_type.
        goal_type = self._get_current_goal_type()
        if not goal_type:
            return DecisionType.TASK_COMPLETED

        return GOAL_TYPE_TO_DECISION.get(goal_type, DecisionType.TASK_COMPLETED)

    def _build_decision_context(self, goal_type: Optional[str]) -> Dict[str, Any]:
        """Build context data for a decision request.

        Args:
            goal_type: Current goal type string.

        Returns:
            Dictionary containing goal_type and other context information.
        """
        scene_graph: AbstractSceneGraph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        goal_id = goal.get("id") if goal else None

        context: Dict[str, Any] = {
            "goal_type": goal_type or "",
            "goal_id": goal_id or "",
        }

        # Merge search context if available.
        goal_monitor = self.world_model_layer.goal_progress_monitor
        if goal_monitor and goal_id:
            search_ctx = goal_monitor.get_search_context_for_decision(goal_id)
            context.update(search_ctx)

        return context

    def _needs_hitl_decision(self) -> bool:
        """Determine whether a HITL decision is currently needed.

        Two cases need a decision:
        1. Skill list execution finished and search did not find the target.
        2. The full task dependency graph has finished, with no pending tasks.

        Returns:
            True if a HITL decision is needed.
        """
        scene_graph: AbstractSceneGraph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        goal_id = goal.get("id") if goal else None

        # Case 1: search did not find the target.
        goal_monitor = self.world_model_layer.goal_progress_monitor
        if goal_id and goal_monitor and goal_monitor.has_search_not_found(goal_id):
            return True

        # Case 2: the task graph is fully executed, with no pending tasks.
        tgm = self.planning_layer.task_graph_manager
        if tgm is not None and not tgm.has_pending_tasks():
            return True

        return False

    async def _request_hitl_decision(self) -> Optional[Dict[str, Any]]:
        """Shared HITL decision request.

        Send a decision request only when:
        1. Skill list execution finished and search did not find the target.
        2. The full task dependency graph has finished.

        Returns:
            Decision result dictionary, or None if no decision is needed.
        """
        if not self.hitl_manager or not self.hitl_manager.is_decision_enabled:
            return None

        if not self._needs_hitl_decision():
            return None

        decision_type = self._determine_decision_type()
        if not decision_type:
            return None

        description = DECISION_DESCRIPTIONS.get(
            decision_type, DECISION_DESCRIPTIONS[DecisionType.TASK_COMPLETED]
        )
        goal_type = self._get_current_goal_type()
        context = self._build_decision_context(goal_type)

        try:
            decision = await self.hitl_manager.request_decision(
                decision_type, context, description=description
            )

            dlog(f"HITL decision received: {decision.get('decision')}",
                 logger=self.logger, level='info')

            return decision
        except Exception as e:
            dlog(f"Error requesting HITL decision: {e}", logger=self.logger, level='error')
            return None
