"""Event configuration module.

Contains definitions and configurations for all event types.
Separates event definitions from the events module for a cleaner architecture.
"""

from .event_types import (
    EventType,
    Event,
    SystemEvent,
    TaskEvent,
    UserEvent,
    ReplyEvent,
    DataModificationEvent,
    DataModificationReplyEvent,
    NewCaseEvent,
    SkillExecutionErrorEvent
)

__all__ = [
    'EventType',
    'Event',
    'SystemEvent',
    'TaskEvent',
    'UserEvent',
    'ReplyEvent',
    'DataModificationEvent',
    'DataModificationReplyEvent',
    'NewCaseEvent',
    'SkillExecutionErrorEvent'
]
