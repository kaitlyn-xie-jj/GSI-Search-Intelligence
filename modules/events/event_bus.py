"""Global Event Bus Implementation

Provides async event communication mechanism between system components.
Based on refactored_system's event_bus implementation with global singleton pattern.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import uuid
import logging

from modules.config.events import Event, ReplyEvent

logger = logging.getLogger(__name__)

# Global event bus instance
_global_event_bus: Optional['EventBus'] = None

# Event handler type definitions
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], asyncio.Future]


class EventSubscription:
    """Event Subscription
    
    Manages information and processing logic for a single event subscription.
    """
    
    def __init__(self, 
                 event_type: str,
                 handler: Union[EventHandler, AsyncEventHandler],
                 subscriber_id: str,
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 priority: int = 0,
                 max_retries: int = 0,
                 retry_delay: float = 1.0):
        """Initialize event subscription
        
        Args:
            event_type: Event type
            handler: Event handler
            subscriber_id: Subscriber ID
            filter_func: Event filter function
            priority: Priority (higher number = higher priority)
            max_retries: Maximum retry count
            retry_delay: Retry delay (seconds)
        """
        self.event_type = event_type
        self.handler = handler
        self.subscriber_id = subscriber_id
        self.filter_func = filter_func
        self.priority = priority
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.subscription_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.call_count = 0
        self.error_count = 0
        self.last_called = None
        self.last_error = None
    
    async def handle_event(self, event: Event) -> None:
        """Handle event (with retry mechanism)
        
        Args:
            event: Event to handle
        """
        retries = 0
        
        while retries <= self.max_retries:
            try:
                self.call_count += 1
                self.last_called = datetime.now()
                
                if asyncio.iscoroutinefunction(self.handler):
                    await self.handler(event)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.handler, event)
                
                # Successfully handled, exit retry loop
                break
                
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                
                if retries < self.max_retries:
                    retries += 1
                    logger.warning(
                        f"Handler {self.subscription_id} failed (attempt {retries}/{self.max_retries + 1}): {e}"
                    )
                    await asyncio.sleep(self.retry_delay * retries)  # Exponential backoff
                else:
                    logger.error(
                        f"Handler {self.subscription_id} failed after {self.max_retries + 1} attempts: {e}",
                        exc_info=True
                    )
                    raise
    
    def matches(self, event: Event) -> bool:
        """Check if event matches subscription
        
        Args:
            event: Event to check
            
        Returns:
            Whether it matches
        """
        if event.event_type != self.event_type:
            return False
        
        if self.filter_func and not self.filter_func(event):
            return False
        
        return True


class EventBus:
    """Event Bus
    
    Provides async event publish and subscribe mechanism with event filtering, history, and statistics.
    """
    
    def __init__(self, max_queue_size: int = 1000, max_concurrent_handlers: int = 100, use_priority_queue: bool = True):
        """Initialize event bus
        
        Args:
            max_queue_size: Maximum event queue size
            max_concurrent_handlers: Maximum concurrent handler count
            use_priority_queue: Whether to use priority queue (default True)
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent_handlers = max_concurrent_handlers
        self.use_priority_queue = use_priority_queue
        
        # Select queue type based on config
        if use_priority_queue:
            self.event_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        else:
            self.event_queue = asyncio.Queue(maxsize=max_queue_size)
            
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.event_history = deque(maxlen=1000)
        
        # Add semaphore to control concurrent handler count
        self._handler_semaphore = asyncio.Semaphore(max_concurrent_handlers)
        
        self.running = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # Enhanced statistics
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscriptions': 0,
            'handler_timeouts': 0,
            'queue_overflows': 0
        }
        
        # Add performance monitoring
        self._processing_times: deque = deque(maxlen=100)
        self._last_health_check = datetime.now()
    
    async def _handle_event_with_timeout(self, event: Event, timeout: float = 30.0) -> None:
        """Handle event with timeout
        
        Args:
            event: Event to handle
            timeout: Timeout in seconds
        """
        start_time = datetime.now()
        
        try:
            # Add to history
            self.event_history.append(event)
            
            # Get matching subscriptions, sorted by priority
            matching_subscriptions = []
            for subscription in self.subscriptions.get(event.event_type, []):
                if subscription.matches(event):
                    matching_subscriptions.append(subscription)
            
            # Sort by priority (higher priority first)
            matching_subscriptions.sort(key=lambda x: x.priority, reverse=True)
            
            # Use semaphore to control concurrent processing
            if matching_subscriptions:
                async def handle_with_semaphore(subscription: EventSubscription) -> None:
                    async with self._handler_semaphore:
                        try:
                            await asyncio.wait_for(
                                subscription.handle_event(event), 
                                timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Handler timeout for subscription {subscription.subscription_id}"
                            )
                            self.stats['handler_timeouts'] += 1
                        except Exception as e:
                            logger.error(
                                f"Handler error for subscription {subscription.subscription_id}: {e}",
                                exc_info=True
                            )
                
                # Process all matching subscriptions concurrently
                tasks = [
                    handle_with_semaphore(subscription)
                    for subscription in matching_subscriptions
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self._processing_times.append(processing_time)
            
            self.stats['events_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error handling event {event.event_id}: {e}", exc_info=True)
            self.stats['events_failed'] += 1
    
    async def publish_batch(self, events: List[Event]) -> None:
        """Publish events in batch
        
        Args:
            events: List of events to publish
        """
        if not self.running:
            logger.warning("Event bus not running, events ignored")
            return
        
        successful = 0
        failed = 0
        
        for event in events:
            try:
                await self.event_queue.put(event)
                successful += 1
            except asyncio.QueueFull:
                logger.error(f"Event queue full, dropping event {event.event_id}")
                failed += 1
                self.stats['queue_overflows'] += 1
        
        self.stats['events_published'] += successful
        self.stats['events_failed'] += failed
        
        logger.debug(f"Batch published: {successful} successful, {failed} failed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics
        
        Returns:
            Performance metrics dict
        """
        if self._processing_times:
            avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            max_processing_time = max(self._processing_times)
            min_processing_time = min(self._processing_times)
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
        
        return {
            'average_processing_time': avg_processing_time,
            'max_processing_time': max_processing_time,
            'min_processing_time': min_processing_time,
            'active_handlers': self.max_concurrent_handlers - self._handler_semaphore._value,
            'queue_utilization': self.event_queue.qsize() / self.max_queue_size,
            'last_health_check': self._last_health_check.isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check
        
        Returns:
            Health status dict
        """
        self._last_health_check = datetime.now()
        
        health_status = {
            'status': 'healthy' if self.running else 'stopped',
            'queue_size': self.event_queue.qsize(),
            'queue_capacity': self.max_queue_size,
            'active_subscriptions': self.stats['active_subscriptions'],
            'processor_running': self.processor_task is not None and not self.processor_task.done(),
            'performance_metrics': self.get_performance_metrics()
        }
        
        # Check if queue is near capacity
        if self.event_queue.qsize() > self.max_queue_size * 0.8:
            health_status['warnings'] = ['Queue utilization high']
        
        # Check for excessive timeouts
        if self.stats['handler_timeouts'] > 10:
            health_status.setdefault('warnings', []).append('High handler timeout rate')
        
        return health_status
    
    async def start(self) -> None:
        """Start event bus"""
        if self.running:
            return
        
        self.running = True
        self.processor_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self) -> None:
        """Stop event bus"""
        if not self.running:
            return
        
        self.running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    def subscribe(self, 
                 event_type: str,
                 handler: Union[EventHandler, AsyncEventHandler],
                 subscriber_id: str,
                 filter_func: Optional[Callable[[Event], bool]] = None,
                 priority: int = 0,
                 max_retries: int = 0,
                 retry_delay: float = 1.0) -> str:
        """Subscribe to event
        
        Args:
            event_type: Event type
            handler: Event handler
            subscriber_id: Subscriber ID
            filter_func: Event filter function
            priority: Priority (higher number = higher priority)
            max_retries: Maximum retry count
            retry_delay: Retry delay (seconds)
            
        Returns:
            Subscription ID
        """
        subscription = EventSubscription(
            event_type, handler, subscriber_id, filter_func,
            priority, max_retries, retry_delay
        )
        self.subscriptions[event_type].append(subscription)
        self.stats['active_subscriptions'] += 1
        
        logger.debug(f"Subscribed to {event_type}: {subscriber_id} (priority: {priority})")
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            Whether unsubscribe succeeded
        """
        for event_type, subscriptions in self.subscriptions.items():
            for i, subscription in enumerate(subscriptions):
                if subscription.subscription_id == subscription_id:
                    del subscriptions[i]
                    self.stats['active_subscriptions'] -= 1
                    logger.debug(f"Unsubscribed: {subscription_id}")
                    return True
        
        return False
    
    def unsubscribe_all(self, subscriber_id: str) -> int:
        """Unsubscribe all for a subscriber
        
        Args:
            subscriber_id: Subscriber ID
            
        Returns:
            Number of unsubscribed subscriptions
        """
        count = 0
        for event_type, subscriptions in self.subscriptions.items():
            original_count = len(subscriptions)
            subscriptions[:] = [
                sub for sub in subscriptions 
                if sub.subscriber_id != subscriber_id
            ]
            removed = original_count - len(subscriptions)
            count += removed
            self.stats['active_subscriptions'] -= removed
        
        if count > 0:
            logger.debug(f"Unsubscribed all for {subscriber_id}: {count} subscriptions")
        
        return count
    
    async def publish(self, event: Event) -> None:
        """Publish event
        
        Args:
            event: Event to publish
        """
        if not self.running:
            logger.warning("Event bus not running, event ignored")
            return
        
        try:
            await self.event_queue.put(event)
            self.stats['events_published'] += 1
            logger.debug(f"Event published: {event.event_type} - {event.event_id}")
            
        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event")
            self.stats['events_failed'] += 1
    
    async def publish_sync(self, event: Event) -> None:
        """Publish event synchronously (immediate processing)
        
        Args:
            event: Event to publish
        """
        await self._handle_event(event)
    
    async def _process_events(self) -> None:
        """Event processing loop"""
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                self.stats['events_failed'] += 1
    
    async def _handle_event(self, event: Event) -> None:
        """Handle single event
        
        Args:
            event: Event to handle
        """
        # Use timeout-based handling logic
        await self._handle_event_with_timeout(event)
    
    def get_event_history(self, 
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Event]:
        """Get event history
        
        Args:
            event_type: Event type filter
            limit: Return count limit
            
        Returns:
            Event list
        """
        events = list(self.event_history)
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics
        
        Returns:
            Statistics dict
        """
        return {
            **self.stats,
            'queue_size': self.event_queue.qsize(),
            'history_size': len(self.event_history),
            'subscription_types': list(self.subscriptions.keys()),
            'use_priority_queue': self.use_priority_queue
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics (alias)
        
        Returns:
            Statistics dict
        """
        return self.get_statistics()
    
    def clear_history(self) -> None:
        """Clear event history"""
        self.event_history.clear()
        logger.info("Event history cleared")
    
    async def publish_high_priority(self, event: Event) -> None:
        """Publish high priority event
        
        Args:
            event: Event to publish (will be set to high priority)
        """
        event.priority = 0  # Set to highest priority
        await self.publish(event)
    
    async def publish_reply(self, request_id: str, reply_to: str, 
                           success: bool = True, result: Dict[str, Any] = None,
                           error_message: str = "") -> None:
        """Publish reply event (highest priority)
        
        Args:
            request_id: Original request ID
            reply_to: Reply target
            success: Whether operation succeeded
            result: Reply result data
            error_message: Error message
        """
        reply_event = ReplyEvent(
            request_id=request_id,
            reply_to=reply_to,
            success=success,
            result=result or {},
            error_message=error_message,
            priority=0  # Highest priority
        )
        await self.publish(reply_event)
    
    def get_priority_distribution(self) -> Dict[int, int]:
        """Get priority distribution in event history
        
        Returns:
            Priority distribution dict {priority: count}
        """
        priority_counts = {}
        for event in self.event_history:
            priority = getattr(event, 'priority', 10)  # Default priority 10
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        return priority_counts


def get_global_event_bus() -> EventBus:
    """Get global event bus instance
    
    Returns:
        Global event bus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_global_event_bus(event_bus: EventBus) -> None:
    """Set global event bus instance
    
    Args:
        event_bus: Event bus instance
    """
    global _global_event_bus
    _global_event_bus = event_bus


async def start_global_event_bus() -> None:
    """Start global event bus"""
    event_bus = get_global_event_bus()
    await event_bus.start()


async def stop_global_event_bus() -> None:
    """Stop global event bus"""
    event_bus = get_global_event_bus()
    await event_bus.stop()


async def publish_event(event: Event) -> None:
    """Publish event to global event bus
    
    Args:
        event: Event to publish
    """
    event_bus = get_global_event_bus()
    await event_bus.publish(event)

async def publish_event_sync(event: Event) -> None:
    """Publish event to global event bus and process immediately
    
    Args:
        event: Event to publish
    """
    event_bus = get_global_event_bus()
    await event_bus.publish_sync(event)


async def publish_high_priority_event(event: Event) -> None:
    """Publish high priority event to global event bus
    
    Args:
        event: Event to publish (will be set to high priority)
    """
    event_bus = get_global_event_bus()
    await event_bus.publish_high_priority(event)


async def publish_reply_event(request_id: str, reply_to: str, 
                             success: bool = True, result: Dict[str, Any] = None,
                             error_message: str = "") -> None:
    """Publish reply event to global event bus (highest priority)
    
    Args:
        request_id: Original request ID
        reply_to: Reply target
        success: Whether operation succeeded
        result: Reply result data
        error_message: Error message
    """
    event_bus = get_global_event_bus()
    await event_bus.publish_reply(request_id, reply_to, success, result, error_message)


def subscribe_event(event_type: str,
                   handler: Union[EventHandler, AsyncEventHandler],
                   subscriber_id: str,
                   filter_func: Optional[Callable[[Event], bool]] = None,
                   priority: int = 0,
                   max_retries: int = 0,
                   retry_delay: float = 1.0) -> str:
    """Subscribe to global event bus events
    
    Args:
        event_type: Event type
        handler: Event handler
        subscriber_id: Subscriber ID
        filter_func: Event filter function
        priority: Priority (higher number = higher priority)
        max_retries: Maximum retry count
        retry_delay: Retry delay (seconds)
        
    Returns:
        Subscription ID
    """
    event_bus = get_global_event_bus()
    return event_bus.subscribe(event_type, handler, subscriber_id, filter_func, priority, max_retries, retry_delay)


def unsubscribe_event(subscription_id: str) -> bool:
    """Unsubscribe from global event bus events
    
    Args:
        subscription_id: Subscription ID
        
    Returns:
        Whether unsubscribe succeeded
    """
    event_bus = get_global_event_bus()
    return event_bus.unsubscribe(subscription_id)