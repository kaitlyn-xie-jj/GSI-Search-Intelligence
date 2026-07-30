# -*- coding: utf-8 -*-
"""SPINE prompt templates."""

from .spine_prompt import build_spine_system_prompt, build_spine_user_prompt
from .spine_examples import SPINE_FEW_SHOT_EXAMPLES
from .spine_observation import format_spine_observation

__all__ = [
    "build_spine_system_prompt",
    "build_spine_user_prompt",
    "SPINE_FEW_SHOT_EXAMPLES",
    "format_spine_observation",
]
