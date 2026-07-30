"""Event Type Definitions.

Defines all event types and related enumerations used in the system.
Separated from the events.event_bus module for cleaner configuration management.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Event type enumeration.
    
    Simplified event type hierarchy containing core event types:
    - TASK: Task-level events (business logic events for running code) - highest priority
    - SYSTEM: System-level events (debug and monitoring info for developers) - medium priority
    - USER: User-level events (UI and interaction events for end users) - lowest priority
    - DATA_MODIFICATION: Data modification events
    - DATA_MODIFICATION_REPLY: Data modification reply events
    - NEW_CASE: New case events
    - SKILL_EXECUTION_ERROR: Skill execution error events
    """
    TASK = "task"                                    # Priority: 1 (highest)
    SYSTEM = "system"                                # Priority: 5 (medium)
    USER = "user"                                    # Priority: 10 (lowest)
    DATA_MODIFICATION = "data_modification"
    DATA_MODIFICATION_REPLY = "data_modification_reply"
    NEW_CASE = "new_case"
    SKILL_EXECUTION_ERROR = "skill_execution_error"

    ROBOT = "robot"
    PROP = "prop"
    SKILL = "skill"
    OBJECT = "object"
    GOAL = "goal"
    CONFIG = "config"
    MONITOR = "monitor"
    ERROR = "error"


@dataclass
class Event(ABC):
    """Event base class.
    
    All events should inherit from this base class and implement the event_type property.
    
    Attributes:
        event_id: Unique event identifier
        timestamp: Event timestamp
        source: Event source
        priority: Event priority (lower number = higher priority, 0 is highest)
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    priority: int = 10  # Default priority is 10
    
    def __lt__(self, other) -> bool:
        """Priority comparison method for PriorityQueue sorting.
        
        Lower number means higher priority; timestamp is used as a tiebreaker.
        
        Args:
            other: Another event object
            
        Returns:
            bool: Whether the current event has higher priority
        """
        if not isinstance(other, Event):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        # When priorities are equal, sort by timestamp (earlier first)
        return self.timestamp < other.timestamp
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Event type.
        
        Returns:
            str: Event type string
        """
        pass


@dataclass
class SystemEvent(Event):
    """System event.
    
    Used for system-level event notifications such as startup, shutdown,
    configuration changes, and debug information.
    Provides system monitoring and debug info for developers.
    
    Attributes:
        message: Event message
        data: Event data
        component: Component name
        operation: Operation type
    """
    
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    component: str = ""
    operation: str = ""
    priority: int = field(default=5)  # System events have medium priority
    
    @property
    def event_type(self) -> str:
        """Return the system event type.
        
        Returns:
            str: System event type identifier
        """
        return EventType.SYSTEM.value


@dataclass
class TaskEvent(Event):
    """Task event.
    
    Used for task-related event notifications, including all business logic events.
    Core business logic events for running code, with the highest priority.
    
    Attributes:
        task_id: Task ID
        action: Action type (created, started, completed, failed, cancelled)
        data: Event data
        entity_type: Entity type (task, robot, skill, object, goal, etc.)
        entity_id: Entity ID
    """
    
    task_id: str = ""
    action: str = ""  # created, started, completed, failed, cancelled
    data: Dict[str, Any] = field(default_factory=dict)
    entity_type: str = ""  # task, robot, skill, object, goal, etc.
    entity_id: str = ""
    priority: int = field(default=1)  # Task events have the highest priority
    
    @property
    def event_type(self) -> str:
        """Return the task event type.
        
        Returns:
            str: Task event type identifier
        """
        return EventType.TASK.value


@dataclass
class UserEvent(Event):
    """User event.
    
    Used for user interface and interaction related event notifications.
    UI and interaction events for end users, with the lowest priority.
    
    Attributes:
        title: Event title
        message: Event message
        icon: Icon type
        action_type: Interaction type (notification, alert, confirmation, etc.)
        ui_data: UI-related data
    """
    
    title: str = ""
    message: str = ""
    icon: str = "info"  # info, warning, error, success
    action_type: str = "notification"  # notification, alert, confirmation, etc.
    ui_data: Dict[str, Any] = field(default_factory=dict)
    priority: int = field(default=10)  # User events have the lowest priority
    
    @property
    def event_type(self) -> str:
        """Return the user event type.
        
        Returns:
            str: User event type identifier
        """
        return EventType.USER.value


@dataclass
class ReplyEvent(Event):
    """Generic reply event.
    
    Used for high-priority reply events in request-reply patterns.
    These events are placed at the front of the queue for fast response.
    A special type of task-level event.
    
    Supports two modes:
    1. Targeted reply: specifies a reply_to target; only specific subscribers receive it
    2. Broadcast reply: reply_to is empty or "*"; all relevant subscribers receive it
    
    Attributes:
        request_id: Original request ID
        reply_to: Reply target (subscriber ID or event ID), optional, supports broadcast mode
        success: Whether the operation succeeded
        result: Reply result data
        error_message: Error message (if failed)
        broadcast: Whether in broadcast mode (auto-set to True when reply_to is empty or "*")
    """
    
    request_id: str = ""
    reply_to: Optional[str] = None  # Optional, supports broadcast mode
    success: bool = True
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    priority: int = field(default=0)  # Highest priority
    
    def __post_init__(self) -> None:
        """Post-initialization: set the broadcast mode flag."""
        # If reply_to is empty, None, or "*", use broadcast mode
        self.broadcast = not self.reply_to or self.reply_to == "*"
    
    @property
    def event_type(self) -> str:
        """Return the reply event type.
        
        Returns:
            str: Task event type identifier (reply events are task-level events)
        """
        return EventType.TASK.value
    
    def is_broadcast(self) -> bool:
        """Check whether this is a broadcast mode reply.
        
        Returns:
            bool: Whether in broadcast mode
        """
        return getattr(self, 'broadcast', False)
    
    def is_targeted_to(self, target_id: str) -> bool:
        """Check whether this reply targets a specific recipient.
        
        Args:
            target_id: Target ID
            
        Returns:
            bool: Whether targeted to this recipient
        """
        if self.is_broadcast():
            return True
        return self.reply_to == target_id


@dataclass
class DataModificationReplyEvent(ReplyEvent):
    """Data modification reply event.
    
    Inherits from ReplyEvent, specifically for replies to data modification operations.
    Supports both broadcast and targeted reply modes.
    
    Attributes:
        operation: Operation type ('add', 'update', 'remove')
        entity_type: Entity type ('node', 'edge')
        entity_id: Entity ID
    """
    
    operation: str = ""
    entity_type: str = ""
    entity_id: str = ""
    
    @property
    def event_type(self) -> str:
        """Return the data modification reply event type.
        
        Returns:
            str: Data modification reply event type identifier
        """
        return EventType.DATA_MODIFICATION_REPLY.value


@dataclass
class DataModificationEvent(Event):
    """Data modification event.
    
    Used to notify node and edge add/update/remove operations. All data modifications
    to SemanticSceneGraph must go through this event to ensure data consistency
    and an event-driven architecture.
    
    Attributes:
        operation: Operation type ('add', 'update', 'remove')
        entity_type: Entity type ('node', 'edge')
        entity_id: Entity ID (for edges, use 'source_id->target_id' format)
        data: Entity data
        request_id: Request ID for tracking and replies
        priority: Event priority (default 1, high priority)
    """
    
    operation: str = ""  # add, update, remove
    entity_type: str = ""  # node, edge
    entity_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    priority: int = 1  # High priority
    
    @property
    def event_type(self) -> str:
        """Return the data modification event type identifier.
        
        Returns:
            str: Data modification event type identifier
        """
        return EventType.DATA_MODIFICATION.value


@dataclass
class NewCaseEvent(Event):
    """New case event.
    
    Used to notify about new case (unexpected situation) events.
    
    Attributes:
        case_id: New case ID
        description: New case description
        entity_id: Entity ID
        entity_type: Entity type
    """
    
    case_id: str = ""
    description: str = ""
    entity_id: str = ""
    entity_type: str = ""
    priority: int = field(default=0)  # New case event priority
    
    @property
    def event_type(self) -> str:
        """Return the new case event type.
        
        Returns:
            str: New case event type identifier
        """
        return EventType.NEW_CASE.value


@dataclass
class SkillExecutionErrorEvent(Event):
    """Skill execution error event.
    
    Used to notify about errors during skill execution, triggering the
    replanning mechanism. This is the only skill-related event type
    that can trigger replanning.
    
    Attributes:
        skill_id: Skill ID
        skill_name: Skill name
        error_type: Error type (timeout, execution_failed, invalid_params, etc.)
        error_message: Detailed error message
        robot_id: ID of the robot executing the skill
        task_id: Related task ID
        execution_context: Execution context information
        retry_count: Number of retries
        can_retry: Whether retry is possible
    """
    
    skill_id: str = ""
    skill_name: str = ""
    error_type: str = ""  # timeout, execution_failed, invalid_params, etc.
    error_message: str = ""
    robot_id: str = ""
    task_id: str = ""
    execution_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    can_retry: bool = False
    priority: int = field(default=0)  # Skill execution error events have the highest priority
    
    @property
    def event_type(self) -> str:
        """Return the skill execution error event type.
        
        Returns:
            str: Skill execution error event type identifier
        """
        return EventType.SKILL_EXECUTION_ERROR.value
