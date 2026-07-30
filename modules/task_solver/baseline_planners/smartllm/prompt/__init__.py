# -*- coding: utf-8 -*-
"""SmartLLM prompt templates."""

from .smartllm_prompt import (
    build_decomposition_system_prompt,
    build_decomposition_user_prompt,
    build_coalition_system_prompt,
    build_coalition_user_prompt,
    build_allocation_system_prompt,
    build_allocation_user_prompt,
)
from .smartllm_examples import (
    DECOMPOSITION_FEW_SHOT_EXAMPLES,
    COALITION_FEW_SHOT_EXAMPLES,
    ALLOCATION_FEW_SHOT_EXAMPLES,
)

__all__ = [
    "build_decomposition_system_prompt",
    "build_decomposition_user_prompt",
    "build_coalition_system_prompt",
    "build_coalition_user_prompt",
    "build_allocation_system_prompt",
    "build_allocation_user_prompt",
    "DECOMPOSITION_FEW_SHOT_EXAMPLES",
    "COALITION_FEW_SHOT_EXAMPLES",
    "ALLOCATION_FEW_SHOT_EXAMPLES",
]
