# -*- coding: utf-8 -*-
"""
Platform Factory - platform factory

Singleton factory responsible for creating and managing platform instances.
Provides unified platform initialization, access, and cleanup interfaces.
"""

import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.abstract_platform_executor import AbstractPlatformExecutor

if TYPE_CHECKING:
    from modules.communication.unified_communicator import UnifiedCommunicator

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Platform type enum."""
    SEMANTIC = "semantic"
    UNREAL = "unreal"


# ==================== Platform Default Boundary Definitions ====================

# Default semantic platform boundary in meters, for 1000x1000 semantic simulation scenes.
SEMANTIC_DEFAULT_BOUNDARY: List[List[float]] = [[-10, -10], [1010, -10], [1010, 1010], [-10, 1010]]

# Default Unreal platform boundary in meters, for UE5 scenes such as Downtown.
UNREAL_DEFAULT_BOUNDARY: List[List[float]] = [[70, 25], [-100, 25], [-70, 125], [70, 125]]


# ==================== Global Singleton Variables ====================

_global_scene_graph: Optional[AbstractSceneGraph] = None
_global_platform_executor: Optional[AbstractPlatformExecutor] = None
_global_communicator: Optional["UnifiedCommunicator"] = None
_current_platform_type: Optional[PlatformType] = None
_platform_lock = asyncio.Lock()


# ==================== Platform Initialization ====================

async def initialize_platform(
    platform_type: PlatformType,
    initial_nodes: Optional[List[Dict[str, Any]]] = None,
    initial_edges: Optional[List[Dict[str, Any]]] = None,
    initial_goal: Optional[Dict[str, Any]] = None,
    **platform_kwargs
) -> AbstractSceneGraph:
    """
    Initialize the platform and create the scene graph instance.
    
    Args:
        platform_type: Platform type, SEMANTIC or UNREAL.
        initial_nodes: Initial node data, required only for the SEMANTIC platform.
        initial_edges: Initial edge data, required only for the SEMANTIC platform.
        initial_goal: Initial goal data.
        **platform_kwargs: Platform-specific parameters.
            - UNREAL: base_url, server_port, timeout, polling_interval
            - UNREAL: hitl_enabled, optional, whether to enable HITL.
    
    Returns:
        AbstractSceneGraph: Scene graph instance.
        
    Raises:
        RuntimeError: If the SEMANTIC platform is missing initial data.
        ValueError: If the platform type is invalid.
    """
    async with _platform_lock:
        global _global_scene_graph, _global_communicator, _current_platform_type
        
        # Clean up first if already initialized.
        if _global_scene_graph is not None:
            logger.warning("Platform already initialized, cleaning up first")
            await _cleanup_platform_internal()
        
        if platform_type == PlatformType.SEMANTIC:
            # Import the semantic platform implementation.
            from modules.platform.semantic_platform.scene_graph_manager import (
                SemanticSceneGraph,
                reset_global_scene_graph,
            )
            
            # Reset singleton state.
            reset_global_scene_graph()
            
            # Create the semantic scene graph.
            _global_scene_graph = SemanticSceneGraph(
                initial_nodes=initial_nodes,
                initial_edges=initial_edges,
                initial_goal=initial_goal,
            )
            logger.info("Semantic platform initialized")
            
        elif platform_type == PlatformType.UNREAL:
            # Import Unreal platform implementation lazily to avoid circular dependencies.
            from modules.platform.unreal_platform.unreal_scene_graph import UnrealSceneGraph
            from modules.communication.unified_communicator import UnifiedCommunicator
            
            # Extract configuration parameters.
            base_url = platform_kwargs.get('base_url', 'http://localhost:8080')
            server_port = platform_kwargs.get('server_port', 8081)
            timeout = platform_kwargs.get('timeout', 30.0)
            hitl_enabled = platform_kwargs.get('hitl_enabled', False)
            shared_communicator = platform_kwargs.get('shared_communicator')
            
            # Use the provided communicator or create a new UnifiedCommunicator singleton.
            if shared_communicator is not None:
                _global_communicator = shared_communicator
                logger.info(f"Using shared communicator: hitl_enabled={hitl_enabled}")
            else:
                _global_communicator = UnifiedCommunicator(
                    unreal_url=base_url,
                    server_port=server_port,
                    timeout=timeout,
                    hitl_enabled=hitl_enabled
                )
                logger.info(f"UnifiedCommunicator created: base_url={base_url}, server_port={server_port}, hitl_enabled={hitl_enabled}")
            
            # Create UnrealSceneGraph with the injected communicator.
            _global_scene_graph = UnrealSceneGraph(
                initial_goal=initial_goal,
                base_url=base_url,
                server_port=server_port,
                timeout=timeout,
                polling_interval=platform_kwargs.get('polling_interval', 0.5),
                hitl_enabled=hitl_enabled,
                communicator=_global_communicator,
            )
            
            # Sync initial state from Unreal.
            await _global_scene_graph.sync_from_unreal()
            logger.info("Unreal platform initialized")
            
        else:
            raise ValueError(f"Unknown platform type: {platform_type}")
        
        _current_platform_type = platform_type
        return _global_scene_graph


# ==================== Instance Access ====================

def get_scene_graph() -> AbstractSceneGraph:
    """Get the global scene graph instance.
    
    Returns:
        AbstractSceneGraph: Scene graph instance.
        
    Raises:
        RuntimeError: If the platform is not initialized.
    """
    global _global_scene_graph
    if _global_scene_graph is None:
        raise RuntimeError("Platform not initialized. Call initialize_platform() first.")
    return _global_scene_graph


def get_platform_executor() -> AbstractPlatformExecutor:
    """Get the global platform executor instance.
    
    Returns:
        AbstractPlatformExecutor: Platform executor instance.
        
    Raises:
        RuntimeError: If the platform executor is not initialized.
    """
    global _global_platform_executor
    if _global_platform_executor is None:
        raise RuntimeError("Platform executor not initialized. Call create_platform_executor() first.")
    return _global_platform_executor


def get_communicator() -> Optional["UnifiedCommunicator"]:
    """Get the global UnifiedCommunicator instance.
    
    Returns:
        UnifiedCommunicator instance, or None if not initialized or not on the UNREAL platform.
    """
    return _global_communicator


def get_platform_type() -> Optional[PlatformType]:
    """Get the current platform type.
    
    Returns:
        Current platform type, or None if not initialized.
    """
    return _current_platform_type


# ==================== Executor Creation ====================

def create_platform_executor(
    scene_graph: AbstractSceneGraph,
    **executor_kwargs
) -> AbstractPlatformExecutor:
    """Create a platform executor instance.
    
    Args:
        scene_graph: Scene graph instance.
        **executor_kwargs: Executor-specific parameters.
        
    Returns:
        AbstractPlatformExecutor: Platform executor instance.
        
    Raises:
        RuntimeError: If the platform type is not set.
    """
    global _global_platform_executor, _global_communicator, _current_platform_type
    
    if _current_platform_type is None:
        raise RuntimeError("Platform type not set. Call initialize_platform() first.")
    
    if _current_platform_type == PlatformType.SEMANTIC:
        from modules.platform.semantic_platform.platform_executor import PlatformExecutor
        _global_platform_executor = PlatformExecutor(
            scene_graph=scene_graph,
            **executor_kwargs
        )
        logger.info("Semantic platform executor created")
        
    elif _current_platform_type == PlatformType.UNREAL:
        from modules.platform.unreal_platform.unreal_platform_executor import UnrealPlatformExecutor
        _global_platform_executor = UnrealPlatformExecutor(
            scene_graph=scene_graph,
            communicator=_global_communicator,
            **executor_kwargs
        )
        logger.info("Unreal platform executor created")
    
    return _global_platform_executor


# ==================== Cleanup and Reset ====================

async def _cleanup_platform_internal() -> None:
    """Internal cleanup function without locking."""
    global _global_scene_graph, _global_platform_executor, _global_communicator, _current_platform_type
    
    if _global_platform_executor:
        try:
            await _global_platform_executor.stop_services()
            _global_platform_executor.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up platform executor: {e}")
        _global_platform_executor = None
    
    if _global_scene_graph:
        try:
            await _global_scene_graph.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up scene graph: {e}")
        _global_scene_graph = None
    
    # Clean up UnifiedCommunicator.
    if _global_communicator:
        try:
            await _global_communicator.close()
        except Exception as e:
            logger.error(f"Error cleaning up communicator: {e}")
        _global_communicator = None
    
    _current_platform_type = None
    logger.info("Platform cleaned up")


async def cleanup_platform() -> None:
    """Clean up platform resources.
    
    Stops all services and releases resources.
    """
    async with _platform_lock:
        await _cleanup_platform_internal()


def reset_platform() -> None:
    """Reset platform state.
    
    Force-resets all global state without cleanup.
    Note: make sure old instance resources have been cleaned up before using this.
    """
    global _global_scene_graph, _global_platform_executor, _global_communicator, _current_platform_type
    _global_scene_graph = None
    _global_platform_executor = None
    _global_communicator = None
    _current_platform_type = None
    logger.info("Platform state reset")


# ==================== Boundary Configuration ====================

def get_default_global_boundary() -> Dict[str, Any]:
    """Get the default global boundary geometry for the current platform.
    
    Returns:
        Boundary geometry dictionary with format: {"kind": "area", "coords": [[x1,y1], ...]}
    """
    if _current_platform_type == PlatformType.UNREAL:
        coords = UNREAL_DEFAULT_BOUNDARY
    else:
        coords = SEMANTIC_DEFAULT_BOUNDARY
    return {"kind": "area", "coords": [list(c) for c in coords]}
