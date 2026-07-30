"""
Enumeration types for the HITL communication system.

This module defines the core enumerations used throughout the communication layer:
- MessageCategory: Categories of messages (INSTRUCTION, REVIEW, DECISION, PLATFORM)
- MessageDirection: Direction of message flow (PYTHON_TO_UE5, UE5_TO_PYTHON)
- ReviewType: Types of review requests (TASK_GRAPH, SKILL_LIST)
- DecisionType: Types of decision requests (SEARCH_NOT_FOUND, PHOTO_COMPLETED, etc.)
"""

from enum import Enum


class MessageCategory(Enum):
    """
    Message category enumeration.
    
    Defines the different categories of messages that can be exchanged
    between Python and UE5 endpoints.
    """
    INSTRUCTION = "instruction"  # User instruction input
    REVIEW = "review"            # Plan review requests/responses
    DECISION = "decision"        # Decision requests/responses
    PLATFORM = "platform"        # Platform-specific messages (skill_list, task_feedback)


class MessageDirection(Enum):
    """
    Message direction enumeration.
    
    Indicates the direction of message flow in the communication system.
    """
    PYTHON_TO_UE5 = "python_to_ue5"  # Outbound from Python to UE5
    UE5_TO_PYTHON = "ue5_to_python"  # Inbound from UE5 to Python


class ReviewType(Enum):
    """
    Review type enumeration.
    
    Defines the types of data that can be sent for operator review.
    """
    TASK_GRAPH = "task_graph"    # Task plan/graph review
    SKILL_LIST = "skill_allocation"   # Skill allocation results review


class DecisionType(Enum):
    """
    Decision type enumeration.
    
    Defines the types of decisions that can be requested from the operator.
    Each value corresponds to a goal type outcome that may require operator input.
    """
    SEARCH_NOT_FOUND = "search_not_found"            # Search completed but target not found
    SEARCH_COMPLETED = "search_completed"            # Search completed
    PHOTO_COMPLETED = "photo_completed"              # Photo taken
    FOLLOW_COMPLETED = "follow_completed"            # Follow completed
    TRANSPORT_COMPLETED = "transport_completed"      # Transport completed
    GUIDE_COMPLETED = "guide_completed"              # Guide completed
    BROADCAST_COMPLETED = "broadcast_completed"      # Verbal broadcast completed
    PATROL_COMPLETED = "patrol_completed"            # Patrol completed
    ASSEMBLY_COMPLETED = "assembly_completed"        # Assembly completed
    ENFORCEMENT_COMPLETED = "enforcement_completed"  # Traffic enforcement completed
    EVIDENCE_COMPLETED = "evidence_completed"        # Evidence collection completed
    EMERGENCY_COMPLETED = "emergency_completed"      # Emergency response completed
    TASK_COMPLETED = "task_completed"                # Generic fallback for unrecognized goal types
