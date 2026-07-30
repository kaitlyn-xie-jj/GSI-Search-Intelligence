# -*- coding: utf-8 -*-
"""
Common components shared across all baseline methods.

This module provides shared prompt components, utilities, and definitions
that are used by multiple baseline planning methods (LLaMAR, SPINE, etc.).
"""

from .action_converter import ActionConverter
from ...sgi_planner.base_feedback_processor import BaseFeedbackProcessor, ReplanningStrategy
from .prompt_components import (
    extract_json,
    build_notes_section,
    PARAMETERIZATION_RULES,
    ADDITIONAL_NOTES,
)
