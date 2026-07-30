#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event System Test Suite

Tests event bus and event manager functionality, including:
1. Basic event subscription and publishing
2. Multiple subscribers handling same event
3. Event processing latency testing
4. Priority handling
5. Error handling and retry mechanism
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import event system components
from modules.events.event_bus import (
    EventBus,
    subscribe_event, publish_event,
    get_global_event_bus, start_global_event_bus, stop_global_event_bus
)
from modules.events.event_manager import (
    EventManager, get_event_manager, start_event_system, stop_event_system,
    event_system_context
)
from modules.config.events import (
    Event, SystemEvent, TaskEvent, EventType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventTestSubscriber:
    """Event Test Subscriber
    
    Class for testing event subscription and handling.
    """
    
    def __init__(self, subscriber_id: str, processing_delay: float = 0.0):
        """
        Initialize test subscriber
        
        Args:
            subscriber_id: Subscriber ID
            processing_delay: Processing delay time (seconds)
        """
        self.subscriber_id = subscriber_id
        self.processing_delay = processing_delay
        self.received_events: List[Event] = []
        self.processing_times: List[float] = []
        self.error_count = 0
        
    async def handle_system_event(self, event: SystemEvent) -> None:
        """Handle system event
        
        Args:
            event: System event
        """
        start_time = time.time()
        
        try:
            # Simulate processing delay
            if self.processing_delay > 0:
                await asyncio.sleep(self.processing_delay)
            
            # Record event
            self.received_events.append(event)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(
                f"[{self.subscriber_id}] Processed system event: {event.message} "
                f"(processing time: {processing_time:.3f}s)"
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.subscriber_id}] Error processing event: {e}")
            raise
    
    async def handle_task_event(self, event: TaskEvent) -> None:
        """Handle task event
        
        Args:
            event: Task event
        """
        start_time = time.time()
        
        try:
            # Simulate processing delay
            if self.processing_delay > 0:
                await asyncio.sleep(self.processing_delay)
            
            # Record event
            self.received_events.append(event)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(
                f"[{self.subscriber_id}] Processed task event: {event.task_id} - {event.action} "
                f"(processing time: {processing_time:.3f}s)"
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.subscriber_id}] Error processing event: {e}")
            raise
    
    async def handle_robot_event(self, event: RobotEvent) -> None:
        """Handle robot event
        
        Args:
            event: Robot event
        """
        start_time = time.time()
        
        try:
            # Simulate processing delay
            if self.processing_delay > 0:
                await asyncio.sleep(self.processing_delay)
            
            # Record event
            self.received_events.append(event)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(
                f"[{self.subscriber_id}] Processed robot event: {event.robot_id} - {event.action} "
                f"(processing time: {processing_time:.3f}s)"
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.subscriber_id}] Error processing event: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics
        
        Returns:
            Statistics dict
        """
        return {
            "subscriber_id": self.subscriber_id,
            "total_events": len(self.received_events),
            "error_count": self.error_count,
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            "max_processing_time": max(self.processing_times) if self.processing_times else 0,
            "min_processing_time": min(self.processing_times) if self.processing_times else 0,
            "processing_delay": self.processing_delay
        }


class ErrorTestSubscriber:
    """Error Test Subscriber
    
    For testing error handling and retry mechanism.
    """
    
    def __init__(self, subscriber_id: str, fail_count: int = 2):
        """
        Initialize error test subscriber
        
        Args:
            subscriber_id: Subscriber ID
            fail_count: Number of initial calls that will fail
        """
        self.subscriber_id = subscriber_id
        self.fail_count = fail_count
        self.call_count = 0
        self.success_count = 0
        self.error_count = 0
    
    async def handle_event_with_errors(self, event: Event) -> None:
        """Handle event (with error simulation)
        
        Args:
            event: Event
        """
        self.call_count += 1
        
        if self.call_count <= self.fail_count:
            self.error_count += 1
            logger.warning(f"[{self.subscriber_id}] Simulated error (call #{self.call_count})")
            raise Exception(f"Simulated error - call count: {self.call_count}")
        else:
            self.success_count += 1
            logger.info(f"[{self.subscriber_id}] Successfully processed event (call #{self.call_count})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics
        
        Returns:
            Statistics dict
        """
        return {
            "subscriber_id": self.subscriber_id,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "fail_count_setting": self.fail_count
        }


async def test_basic_subscription():
    """Test basic event subscription and publishing"""
    logger.info("=== Test Basic Event Subscription and Publishing ===")
    
    # Create subscriber
    subscriber = EventTestSubscriber("basic_subscriber")
    
    # Subscribe to system events
    subscription_id = subscribe_event(
        EventType.SYSTEM.value,
        subscriber.handle_system_event,
        subscriber.subscriber_id
    )
    
    logger.info(f"Subscription ID: {subscription_id}")
    
    # Publish several system events
    events = [
        SystemEvent(message="System started", source="test"),
        SystemEvent(message="Configuration loaded", source="test"),
        SystemEvent(message="Service ready", source="test")
    ]
    
    for event in events:
        await publish_event(event)
        await asyncio.sleep(0.1)  # Brief delay to ensure event processing
    
    # Wait for event processing to complete
    await asyncio.sleep(1.0)
    
    # Output statistics
    stats = subscriber.get_statistics()
    logger.info(f"Basic subscription test stats: {stats}")
    
    return subscription_id, subscriber


async def test_multiple_subscribers():
    """Test multiple subscribers handling same event"""
    logger.info("=== Test Multiple Subscribers ===")
    
    # Create multiple subscribers with different processing delays
    subscribers = [
        EventTestSubscriber("fast_subscriber", 0.1),
        EventTestSubscriber("medium_subscriber", 0.3),
        EventTestSubscriber("slow_subscriber", 0.5)
    ]
    
    subscription_ids = []
    
    # Subscribe each subscriber to task events with different priorities
    for i, subscriber in enumerate(subscribers):
        priority = len(subscribers) - i  # Fast subscriber has higher priority
        subscription_id = subscribe_event(
            EventType.TASK.value,
            subscriber.handle_task_event,
            subscriber.subscriber_id,
            priority=priority
        )
        subscription_ids.append(subscription_id)
        logger.info(f"Subscriber {subscriber.subscriber_id} subscribed, priority: {priority}")
    
    # Publish task events
    task_events = [
        TaskEvent(task_id="task_001", action="started", source="test"),
        TaskEvent(task_id="task_002", action="completed", source="test"),
        TaskEvent(task_id="task_003", action="failed", source="test")
    ]
    
    start_time = time.time()
    
    for event in task_events:
        await publish_event(event)
    
    # Wait for all event processing to complete
    await asyncio.sleep(2.0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Multiple subscribers test total time: {total_time:.3f}s")
    
    # Output statistics for each subscriber
    for subscriber in subscribers:
        stats = subscriber.get_statistics()
        logger.info(f"Subscriber stats - {stats}")
    
    return subscription_ids, subscribers


async def test_event_latency():
    """Test event processing latency"""
    logger.info("=== Test Event Processing Latency ===")
    
    # Create subscriber
    subscriber = EventTestSubscriber("latency_subscriber", 0.2)
    
    # Subscribe to robot events
    subscription_id = subscribe_event(
        EventType.ROBOT.value,
        subscriber.handle_robot_event,
        subscriber.subscriber_id
    )
    
    # Record publish and processing times
    publish_times = []
    event_count = 5
    
    logger.info(f"Publishing {event_count} robot events...")
    
    for i in range(event_count):
        publish_time = time.time()
        publish_times.append(publish_time)
        
        event = RobotEvent(
            robot_id=f"robot_{i:03d}",
            action="status_changed",
            source="test",
            data={"status": "active", "publish_time": publish_time}
        )
        
        await publish_event(event)
        
        # Interval between publishes
        await asyncio.sleep(0.1)
    
    # Wait for all event processing to complete
    await asyncio.sleep(3.0)
    
    # Calculate latency
    latencies = []
    for i, event in enumerate(subscriber.received_events):
        if i < len(publish_times):
            publish_time = publish_times[i]
            # Assume event processing completion time is publish time + processing time
            processing_time = subscriber.processing_times[i]
            latency = processing_time  # Simplified calculation
            latencies.append(latency)
            
            logger.info(
                f"Event {i}: publish time {publish_time:.3f}, "
                f"processing time {processing_time:.3f}s, latency {latency:.3f}s"
            )
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        logger.info(f"Latency stats - Avg: {avg_latency:.3f}s, Max: {max_latency:.3f}s, Min: {min_latency:.3f}s")
    
    return subscription_id, subscriber


async def test_error_handling_and_retry():
    """Test error handling and retry mechanism"""
    logger.info("=== Test Error Handling and Retry Mechanism ===")
    
    # Create error test subscriber
    error_subscriber = ErrorTestSubscriber("error_subscriber", fail_count=2)
    
    # Subscribe to system events with retry parameters
    subscription_id = subscribe_event(
        EventType.SYSTEM.value,
        error_subscriber.handle_event_with_errors,
        error_subscriber.subscriber_id,
        max_retries=3,
        retry_delay=0.5
    )
    
    # Publish a system event
    event = SystemEvent(message="Test error handling", source="test")
    await publish_event(event)
    
    # Wait for retries to complete
    await asyncio.sleep(5.0)
    
    # Output statistics
    stats = error_subscriber.get_statistics()
    logger.info(f"Error handling test stats: {stats}")
    
    return subscription_id, error_subscriber


async def test_event_bus_performance():
    """Test event bus performance"""
    logger.info("=== Test Event Bus Performance ===")
    
    event_bus = get_global_event_bus()
    
    # Get initial statistics
    initial_stats = event_bus.get_statistics()
    logger.info(f"Initial stats: {initial_stats}")
    
    # Create multiple subscribers
    subscribers = [EventTestSubscriber(f"perf_subscriber_{i}", 0.01) for i in range(3)]
    
    # Subscribe to events
    for subscriber in subscribers:
        subscribe_event(
            EventType.SYSTEM.value,
            subscriber.handle_system_event,
            subscriber.subscriber_id
        )
    
    # Batch publish events
    event_count = 20
    start_time = time.time()
    
    events = [
        SystemEvent(message=f"Performance test event {i}", source="perf_test")
        for i in range(event_count)
    ]
    
    for event in events:
        await publish_event(event)
    
    # Wait for processing to complete
    await asyncio.sleep(2.0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Get final statistics
    final_stats = event_bus.get_statistics()
    performance_metrics = event_bus.get_performance_metrics()
    
    logger.info(f"Performance test complete:")
    logger.info(f"  - Total time: {total_time:.3f}s")
    logger.info(f"  - Event count: {event_count}")
    logger.info(f"  - Throughput: {event_count / total_time:.2f} events/s")
    logger.info(f"  - Final stats: {final_stats}")
    logger.info(f"  - Performance metrics: {performance_metrics}")
    
    return subscribers


async def main():
    """Main test function"""
    logger.info("Starting event system tests")
    
    try:
        # Use event system context manager
        async with event_system_context(max_queue_size=1000):
            logger.info("Event system started")
            
            # Run various tests
            await test_basic_subscription()
            await asyncio.sleep(1.0)
            
            await test_multiple_subscribers()
            await asyncio.sleep(1.0)
            
            await test_event_latency()
            await asyncio.sleep(1.0)
            
            await test_error_handling_and_retry()
            await asyncio.sleep(1.0)
            
            await test_event_bus_performance()
            
            # Get final system statistics
            event_manager = get_event_manager()
            final_stats = event_manager.get_statistics()
            logger.info(f"Final system stats: {final_stats}")
            
        logger.info("Event system stopped")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise
    
    logger.info("Event system tests complete")


if __name__ == "__main__":
    # Run tests
    asyncio.run(main())
