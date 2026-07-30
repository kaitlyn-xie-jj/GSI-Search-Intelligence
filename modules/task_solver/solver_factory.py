# -*- coding: utf-8 -*-
"""
Solver Factory

Always creates UnifiedTaskSolver, which internally creates the appropriate
planning layer based on solver_type.
"""
from typing import Any

from modules.config.system_config import config


def create_solver(**kwargs: Any):
    """Create a UnifiedTaskSolver based on config.solver_type.

    Pass solver_type in kwargs to override the default value from config.

    Returns:
        UnifiedTaskSolver instance.
    """
    from modules.task_solver.unified_task_solver import UnifiedTaskSolver

    solver_type = kwargs.pop("solver_type", None) or config.get_config("solver_type", "sgi")
    
    # Get solver config, if any.
    solver_config = config.get_config("solver_config", {}) or {}

    return UnifiedTaskSolver(
        solver_type=solver_type,
        solver_config=solver_config,
        **kwargs,
    )
