"""Global Event Manager

Provides initialization, startup, shutdown, and management for the global event bus.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from .event_bus import (
    EventBus, 
    get_global_event_bus, set_global_event_bus,
    start_global_event_bus, stop_global_event_bus,
)

from modules.config.events import SystemEvent

logger = logging.getLogger(__name__)


class EventManager:
    """Global Event Manager
    
    Manages the lifecycle of the global event bus and provides unified event management interface.
    """
    
    def __init__(self):
        """Initialize event manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._event_bus: Optional[EventBus] = None
        self._is_running = False
        self._startup_tasks: List[asyncio.Task] = []
        self._shutdown_tasks: List[asyncio.Task] = []
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._health_check_interval = 60.0  # 60 second health check interval
        self._initialized = True
        
        # Add graceful shutdown support
        self._shutdown_event = asyncio.Event()
        self._shutdown_timeout = 30.0
        
        logger.info("EventManager initialized")
    
    async def start(self, 
                   max_queue_size: int = 1000, 
                   max_concurrent_handlers: int = 100,
                   enable_health_monitoring: bool = True) -> None:
        """Start event manager
        
        Args:
            max_queue_size: Maximum event queue size
            max_concurrent_handlers: Maximum concurrent handler count
            enable_health_monitoring: Whether to enable health monitoring
        """
        if self._is_running:
            logger.warning("EventManager is already running")
            return
        
        try:
            # Create or get event bus
            if self._event_bus is None:
                self._event_bus = EventBus(
                    max_queue_size=max_queue_size,
                    max_concurrent_handlers=max_concurrent_handlers
                )
                set_global_event_bus(self._event_bus)
            
            # Start event bus
            await start_global_event_bus()
            self._is_running = True
            
            # Start health monitoring
            if enable_health_monitoring:
                self._health_monitor_task = asyncio.create_task(
                    self._health_monitor_loop()
                )
            
            # Execute startup tasks
            if self._startup_tasks:
                startup_results = await asyncio.gather(
                    *self._startup_tasks, 
                    return_exceptions=True
                )
                
                # Check startup task results
                for i, result in enumerate(startup_results):
                    if isinstance(result, Exception):
                        logger.error(f"Startup task {i} failed: {result}")
                
                self._startup_tasks.clear()
            
            # Publish system startup event
            await self._publish_system_event(
                "started", 
                "Event manager started successfully",
                {
                    "max_queue_size": max_queue_size,
                    "max_concurrent_handlers": max_concurrent_handlers,
                    "health_monitoring": enable_health_monitoring
                }
            )
            
            logger.info("EventManager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start EventManager: {e}")
            self._is_running = False
            raise
    
    async def stop(self, timeout: Optional[float] = None) -> None:
        """Gracefully stop event manager
        
        Args:
            timeout: Stop timeout (seconds)
        """
        if not self._is_running:
            logger.warning("EventManager is not running")
            return
        
        stop_timeout = timeout or self._shutdown_timeout
        
        try:
            # Set shutdown event
            self._shutdown_event.set()
            
            # Publish system stop event
            await self._publish_system_event(
                "stopping", 
                "Event manager is stopping"
            )
            
            # Stop health monitoring
            if self._health_monitor_task:
                self._health_monitor_task.cancel()
                try:
                    await asyncio.wait_for(
                        self._health_monitor_task, 
                        timeout=5.0
                    )
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self._health_monitor_task = None
            
            # Wait for all startup tasks to complete
            if self._startup_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._startup_tasks, return_exceptions=True),
                    timeout=stop_timeout / 2
                )
                self._startup_tasks.clear()
            
            # Execute shutdown tasks
            if self._shutdown_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._shutdown_tasks, return_exceptions=True),
                    timeout=stop_timeout / 2
                )
                self._shutdown_tasks.clear()
            
            # Stop event bus
            await stop_global_event_bus()
            self._is_running = False
            
            logger.info("EventManager stopped successfully")
            
        except asyncio.TimeoutError:
            logger.warning(f"EventManager stop timeout after {stop_timeout}s")
            self._is_running = False
        except Exception as e:
            logger.error(f"Failed to stop EventManager: {e}")
            self._is_running = False
            raise
        finally:
            self._shutdown_event.clear()
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop"""
        while self._is_running and not self._shutdown_event.is_set():
            try:
                if self._event_bus:
                    health_status = await self._event_bus.health_check()
                    
                    # Log health status
                    if health_status.get('warnings'):
                        logger.warning(
                            f"Event bus health warnings: {health_status['warnings']}"
                        )
                    
                    # If status is unhealthy, publish warning event
                    if health_status['status'] != 'healthy':
                        await self._publish_system_event(
                            "health_warning",
                            f"Event bus health status: {health_status['status']}",
                            health_status
                        )
                
                # Wait for next check
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._health_check_interval
                )
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue to next check
                continue
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status
        
        Returns:
            Health status dict
        """
        if not self._event_bus:
            return {
                "status": "not_initialized",
                "is_running": self._is_running
            }
        
        return await self._event_bus.health_check()
    
    async def restart(self, max_queue_size: int = 1000) -> None:
        """Restart event manager
        
        Args:
            max_queue_size: Maximum event queue size
        """
        await self.stop()
        await self.start(max_queue_size)
    
    def is_running(self) -> bool:
        """Check if event manager is running
        
        Returns:
            Whether it is running
        """
        return self._is_running
    
    def get_event_bus(self) -> Optional[EventBus]:
        """Get event bus instance
        
        Returns:
            Event bus instance
        """
        return self._event_bus
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event manager statistics
        
        Returns:
            Statistics dict
        """
        if not self._event_bus:
            return {
                "is_running": self._is_running,
                "event_bus_available": False
            }
        
        stats = self._event_bus.get_statistics()
        stats.update({
            "is_running": self._is_running,
            "event_bus_available": True,
            "startup_tasks": len(self._startup_tasks),
            "shutdown_tasks": len(self._shutdown_tasks)
        })
        
        return stats
    
    def add_startup_task(self, coro) -> None:
        """Add startup task
        
        Args:
            coro: Coroutine function
        """
        if self._is_running:
            # If already running, execute task immediately
            task = asyncio.create_task(coro)
            self._startup_tasks.append(task)
        else:
            # If not running, add to startup task list
            self._startup_tasks.append(coro)
    
    def add_shutdown_task(self, coro) -> None:
        """Add shutdown task
        
        Args:
            coro: Coroutine function
        """
        self._shutdown_tasks.append(coro)
    
    async def _publish_system_event(self, action: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Publish system event
        
        Args:
            action: Action type
            message: Event message
            data: Event data
        """
        try:
            if self._event_bus and self._event_bus.running:
                event = SystemEvent(
                    message=message,
                    source="event_manager",
                    data=data or {"action": action}
                )
                await self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to publish system event: {e}")
    
    @asynccontextmanager
    async def managed_lifecycle(self, max_queue_size: int = 1000):
        """Event manager lifecycle context manager
        
        Args:
            max_queue_size: Maximum event queue size
        
        Usage:
            async with event_manager.managed_lifecycle():
                # Use event bus here
                pass
        """
        await self.start(max_queue_size)
        try:
            yield self
        finally:
            await self.stop()


# Global event manager instance
_global_event_manager: Optional[EventManager] = None


def get_event_manager() -> EventManager:
    """Get global event manager instance
    
    Returns:
        Global event manager instance
    """
    global _global_event_manager
    if _global_event_manager is None:
        _global_event_manager = EventManager()
    return _global_event_manager


async def start_event_system(max_queue_size: int = 1000) -> EventManager:
    """Start event system
    
    Args:
        max_queue_size: Maximum event queue size
        
    Returns:
        Event manager instance
    """
    manager = get_event_manager()
    await manager.start(max_queue_size)
    return manager


async def stop_event_system() -> None:
    """Stop event system"""
    manager = get_event_manager()
    await manager.stop()


async def restart_event_system(max_queue_size: int = 1000) -> EventManager:
    """Restart event system
    
    Args:
        max_queue_size: Maximum event queue size
        
    Returns:
        Event manager instance
    """
    manager = get_event_manager()
    await manager.restart(max_queue_size)
    return manager


def is_event_system_running() -> bool:
    """Check if event system is running
    
    Returns:
        Whether it is running
    """
    manager = get_event_manager()
    return manager.is_running()


def get_event_system_statistics() -> Dict[str, Any]:
    """Get event system statistics
    
    Returns:
        Statistics dict
    """
    manager = get_event_manager()
    return manager.get_statistics()


@asynccontextmanager
async def event_system_context(max_queue_size: int = 1000):
    """Event system context manager
    
    Args:
        max_queue_size: Maximum event queue size
    
    Usage:
        async with event_system_context():
            # Use event system here
            pass
    """
    manager = get_event_manager()
    async with manager.managed_lifecycle(max_queue_size):
        yield manager


# Convenience functions
async def publish_system_message(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Publish system message event
    
    Args:
        message: Message content
        data: Event data
    """
    from .event_bus import publish_event
    
    event = SystemEvent(
        message=message,
        source="system",
        data=data or {}
    )
    await publish_event(event)


async def publish_system_error(error_message: str, error_data: Optional[Dict[str, Any]] = None) -> None:
    """Publish system error event
    
    Args:
        error_message: Error message
        error_data: Error data
    """
    await publish_system_message(
        f"System Error: {error_message}",
        {"error": True, "error_data": error_data or {}}
    )


async def publish_system_warning(warning_message: str, warning_data: Optional[Dict[str, Any]] = None) -> None:
    """Publish system warning event
    
    Args:
        warning_message: Warning message
        warning_data: Warning data
    """
    await publish_system_message(
        f"System Warning: {warning_message}",
        {"warning": True, "warning_data": warning_data or {}}
    )


async def publish_system_info(info_message: str, info_data: Optional[Dict[str, Any]] = None) -> None:
    """Publish system info event
    
    Args:
        info_message: Info message
        info_data: Info data
    """
    await publish_system_message(
        f"System Info: {info_message}",
        {"info": True, "info_data": info_data or {}}
    )
