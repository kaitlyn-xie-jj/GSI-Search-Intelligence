"""
Human-in-the-Loop (HITL) Interaction Module.

This module provides the core components for human-in-the-loop interactions
in the multi-robot task planning system. It includes:

- HITLConfig: Configuration dataclass for HITL settings
- InteractionState: State tracking for pending interactions
- HITLManager: Global singleton manager for all HITL operations

Usage:
    from modules.hitl import get_hitl_manager, HITLConfig
    
    # Get the global manager instance
    manager = get_hitl_manager()
    
    # Initialize with configuration
    manager.initialize(config_dict, communicator)
    
    # Use HITL features
    if manager.is_instruction_enabled:
        instruction = await manager.wait_for_instruction()
"""

from modules.hitl.config import HITLConfig, InteractionState
from modules.hitl.hitl_manager import HITLManager, get_hitl_manager

__all__ = [
    'HITLConfig',
    'InteractionState',
    'HITLManager',
    'get_hitl_manager',
]
