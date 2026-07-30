#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Interface Module

This module contains implementations of large language model interfaces, including:
- gpt: GPT model interface
- llm: Generic LLM interface
- model_manager: Model manager
- qwen: Qwen model interface
"""

__version__ = "1.0.0"
__author__ = "SGI Team"

# Import main components
try:
    from .llm import *
    from .model_manager import *
    from .gpt import *
    from .qwen import *
except ImportError:
    # Provide friendly error message if import fails
    import warnings
    warnings.warn("Some LLM interface components could not be imported, please check dependencies")

__all__ = [
    'gpt',
    'llm', 
    'model_manager',
    'qwen'
]