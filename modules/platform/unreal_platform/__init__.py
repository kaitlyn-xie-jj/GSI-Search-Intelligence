# -*- coding: utf-8 -*-
"""
Unreal Platform Module - Unreal Engine platform implementation

This module provides platform implementations for communicating with Unreal Engine:
- UnrealSceneGraph: scene graph implementation that syncs data with Unreal over HTTP
- UnrealPlatformExecutor: platform executor that sends skills to Unreal and handles feedback
- UnrealDataAdapter: data format adapter
"""

from modules.platform.unreal_platform.unreal_scene_graph import UnrealSceneGraph
from modules.platform.unreal_platform.unreal_platform_executor import UnrealPlatformExecutor
from modules.platform.unreal_platform.data_adapters import UnrealDataAdapter
from modules.communication.unified_communicator import UnifiedCommunicator as UnrealHttpClient

__all__ = [
    "UnrealSceneGraph",
    "UnrealPlatformExecutor",
    "UnrealDataAdapter",
    "UnrealHttpClient",
]
