#!/usr/bin/env python3
"""Start Event System Web UI Server

This script launches the Web visualization interface for the event system.
It automatically starts the event manager and UI server.
"""

import asyncio
import logging
import sys
import threading
import time
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from modules.events.event_manager import get_event_manager, event_system_context
from modules.events.event_ui import start_ui_server, get_ui_server
from modules.events.event_bus import (
    get_global_event_bus, SystemEvent, TaskEvent, UserEvent,
    subscribe_event, publish_event
)

class EventUILauncher:
    """Event UI Launcher
    
    Coordinates the startup of the event system and Web UI.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        self.host = host
        self.port = port
        self.debug = debug
        self.event_manager = None
        self.ui_server = None
        self._running = False
        
    async def start_event_system(self) -> None:
        """Start the event system"""
        try:
            logger.info("Starting event system...")
            self.event_manager = get_event_manager()
            await self.event_manager.start(
                max_queue_size=2000,
                max_concurrent_handlers=200,
                enable_health_monitoring=True
            )
            logger.info("Event system started successfully")
            
            await self._create_demo_subscriptions()
            
        except Exception as e:
            logger.error(f"Failed to start event system: {e}")
            raise
    
    async def _create_demo_subscriptions(self) -> None:
        """Create demo subscriptions"""
        try:
            def system_event_handler(event):
                logger.info(f"System event handler: {event.message}")
            
            subscribe_event(
                'system',
                system_event_handler,
                'demo-system-subscriber',
                priority=10
            )
            
            def task_event_handler(event):
                logger.info(f"Task event handler: {event.action} - {event.task_id}")
            
            subscribe_event(
                'task',
                task_event_handler,
                'demo-task-subscriber',
                priority=5
            )
            
            async def user_event_handler(event):
                logger.info(f"User event handler: {event.title} - {event.message}")
                await asyncio.sleep(0.1)
            
            subscribe_event(
                'user',
                user_event_handler,
                'demo-user-subscriber',
                priority=8
            )
            
            logger.info("Demo subscriptions created")
            
            await self._publish_demo_events()
            
        except Exception as e:
            logger.error(f"Failed to create demo subscriptions: {e}")
    
    async def _publish_demo_events(self) -> None:
        """Publish demo events"""
        try:
            await publish_event(SystemEvent(
                message="Event UI system started",
                source="event-ui-launcher",
                data={"version": "1.0.0", "features": ["real-time", "monitoring", "visualization"]}
            ))
            
            await publish_event(TaskEvent(
                task_id="demo-task-001",
                action="created",
                source="event-ui-launcher",
                data={"description": "Demo task", "priority": "normal"}
            ))
            
            await publish_event(UserEvent(
                title="Demo user event",
                message="User action demo",
                source="event-ui-launcher",
                ui_data={"action": "demo", "timestamp": "2024-01-01T00:00:00Z"}
            ))
            
            logger.info("Demo events published")
            
        except Exception as e:
            logger.error(f"Failed to publish demo events: {e}")
    
    def start_ui_server_direct(self) -> None:
        """Start UI server directly"""
        try:
            logger.info(f"Starting Web UI server http://{self.host}:{self.port}")
            start_ui_server(self.host, self.port, self.debug)
        except Exception as e:
            logger.error(f"Failed to start UI server: {e}")
    
    def run(self) -> None:
        """Run the complete event UI system"""
        try:
            self._running = True
            
            asyncio.run(self.start_event_system())
            
            logger.info("="*60)
            logger.info("🎉 Event system UI started!")
            logger.info(f"📊 Web interface: http://{self.host}:{self.port}")
            logger.info("🔧 Features:")
            logger.info("   - Real-time event monitoring")
            logger.info("   - Event statistics")
            logger.info("   - Subscription management")
            logger.info("   - Performance charts")
            logger.info("   - Test event publishing")
            logger.info("   - System health check")
            logger.info("="*60)
            logger.info("Press Ctrl+C to stop the server")
            
            if self.debug:
                def demo_generator():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._demo_event_generator())
                
                demo_thread = threading.Thread(target=demo_generator, daemon=True)
                demo_thread.start()
            
            self.start_ui_server_direct()
            
        except Exception as e:
            logger.error(f"Failed to run event UI system: {e}")
            raise
    
    async def _demo_event_generator(self) -> None:
        """Demo event generator (runs only in debug mode)"""
        counter = 0
        
        while self._running:
            try:
                await asyncio.sleep(5)
                counter += 1
                
                event_type = counter % 4
                
                if event_type == 0:
                    await publish_event(SystemEvent(
                        message=f"Periodic system check #{counter}",
                        source="demo-generator",
                        data={"check_id": counter, "status": "ok"}
                    ))
                elif event_type == 1:
                    await publish_event(TaskEvent(
                        task_id=f"auto-task-{counter:03d}",
                        action="progress",
                        source="demo-generator",
                        data={"progress": (counter * 10) % 100}
                    ))
                elif event_type == 2:
                    await publish_event(RobotEvent(
                        robot_id=f"robot-{(counter % 3) + 1:02d}",
                        action="position_update",
                        source="demo-generator",
                        data={"position": {"x": counter % 20, "y": (counter * 2) % 15}}
                    ))
                else:
                    skills = ["navigation", "manipulation", "perception", "planning"]
                    await publish_event(SkillEvent(
                        skill_name=skills[counter % len(skills)],
                        robot_id=f"robot-{(counter % 2) + 1:02d}",
                        action="completed",
                        source="demo-generator",
                        data={"duration": (counter % 10) + 1}
                    ))
                
            except Exception as e:
                logger.error(f"Failed to generate demo event: {e}")
                await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the event UI system"""
        try:
            self._running = False
            logger.info("Stopping event system...")
            
            if self.event_manager:
                await self.event_manager.stop()
            
            logger.info("Event system stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop event system: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start Event System Web UI')
    parser.add_argument('--host', default='127.0.0.1', help='Server host address')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        import flask
        import flask_socketio
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please run: pip install flask flask-socketio")
        return
    
    launcher = EventUILauncher(args.host, args.port, args.debug)
    try:
        launcher.run()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Program exited abnormally: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
