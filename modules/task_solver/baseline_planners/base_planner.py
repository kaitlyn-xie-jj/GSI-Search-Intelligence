# -*- coding: utf-8 -*-
"""
Baseline Planner Abstract Base Class - Common interface for all baseline planning methods

Defines the standard interface for the planning layer, called by UnifiedTaskSolver.
Control flow methods such as solve_task, initialize, cleanup belong to UnifiedTaskSolver;
BaselinePlanner is only responsible for planning layer logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class BaselinePlanner(ABC):
    """Abstract base class for all baseline planning methods.

    Defines the standard interface for the planning layer, called by UnifiedTaskSolver.
    Subclasses must implement five core methods: generate_plan, process_feedback,
    is_task_completed, reset, get_metrics.

    """

    @abstractmethod
    async def generate_plan(self) -> Optional[Dict]:
        """Generate a plan, returning in dispatcher_result format.

        Subclasses should save a context reference during initialization so it doesn't need to be passed in each call.
        
        Returns:
            A dispatcher_result dictionary compatible with PlanTranslateCoordinator,
            or None indicating planning failure.
        """
        ...

    @abstractmethod
    async def process_feedback(
        self,
        outcomes: List[Dict],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """Process execution feedback and update internal state.

        Args:
            outcomes: List of outcomes returned by the platform execution layer.
            newcase_events: List of unexpected events pushed by EventBus.
            context: WorkflowContext instance.
        """
        ...

    @abstractmethod
    def is_task_completed(self) -> bool:
        """Determine whether the task is completed.

        Returns:
            True indicates the task is completed and the loop should terminate.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state, used before starting a new task."""
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get experiment metrics in a format compatible with MetricsManager.

        Returns:
            Dict[str, Any]: Dictionary containing experiment metrics.
        """
        ...

    def get_goal_config(self) -> Optional[Dict]:
        """Get goal configuration from context._generated_text.

        Returns:
            Optional[Dict]: Goal configuration dictionary, or None if unavailable.
        """
        ctx = getattr(self, '_context', None) or getattr(self, 'context', None)
        if ctx is None:
            return None
        gt = getattr(ctx, '_generated_text', None)
        if gt is None:
            return None
        goal = gt.get('goal')
        if goal:
            return getattr(goal, 'config', None) or goal
        return None

    def get_area_boundaries(self) -> Optional[Dict]:
        """Get area boundary information from context._generated_text.

        Returns:
            Dict: Area boundary dictionary, or empty dict {} if unavailable.
        """
        ctx = getattr(self, '_context', None) or getattr(self, 'context', None)
        if ctx is None:
            return {}
        gt = getattr(ctx, '_generated_text', None)
        if gt is None:
            return {}
        return gt.get('area_boundaries', {}) or {}

    def get_category_map(self) -> Optional[Dict]:
        """Get category mapping information from context._generated_text.

        Returns:
            Dict: Category mapping dictionary, or empty dict {} if unavailable.
        """
        ctx = getattr(self, '_context', None) or getattr(self, 'context', None)
        if ctx is None:
            return {}
        gt = getattr(ctx, '_generated_text', None)
        if gt is None:
            return {}
        return gt.get('category_map', {}) or {}

    def get_runtime_params(self) -> Optional[Dict]:
        """Get runtime parameters from context._generated_text.

        Returns:
            Dict: Runtime parameters dictionary, or empty dict {} if unavailable.
        """
        ctx = getattr(self, '_context', None) or getattr(self, 'context', None)
        if ctx is None:
            return {}
        gt = getattr(ctx, '_generated_text', None)
        if gt is None:
            return {}
        return gt.get('runtime_params', {}) or {}

    def get_dependencies(self) -> Optional[Dict]:
        """Get task dependencies.

        Baseline methods return None by default (no dependencies).
        Subclasses can override this method to provide specific dependencies.

        Returns:
            Optional[Dict]: Task dependency dictionary. Returns None by default.
        """
        return None
