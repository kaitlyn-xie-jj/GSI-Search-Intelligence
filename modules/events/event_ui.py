"""Event System Web UI

Provides visualization interface for the event bus, including real-time monitoring, statistics, subscription management, etc.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import time

from .event_bus import (
    get_global_event_bus, EventBus,
)
from .event_manager import get_event_manager
from modules.config.events import SystemEvent, TaskEvent

logger = logging.getLogger(__name__)


class EventUIServer:
    """Event System UI Server
    
    Provides Web interface to monitor and manage the event system.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5000):
        """Initialize UI server
        
        Args:
            host: Server host address
            port: Server port
        """
        import os
        self.host = host
        self.port = port
        
        # Get absolute path of current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, 'templates')
        
        self.app = Flask(__name__, template_folder=template_dir)
        self.app.config['SECRET_KEY'] = 'event-ui-secret-key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._event_bus: Optional[EventBus] = None
        
        # Real-time data cache
        self._recent_events: List[Dict[str, Any]] = []
        self._max_recent_events = 100
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info(f"EventUIServer initialized on {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template('event_dashboard.html')
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status"""
            try:
                event_manager = get_event_manager()
                event_bus = get_global_event_bus()
                
                status = {
                    'event_manager_running': event_manager.is_running(),
                    'event_bus_stats': event_bus.get_statistics(),
                    'performance_metrics': event_bus.get_performance_metrics(),
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/events/history')
        def get_event_history():
            """Get event history"""
            try:
                event_type = request.args.get('type')
                limit = int(request.args.get('limit', 50))
                
                event_bus = get_global_event_bus()
                events = event_bus.get_event_history(event_type, limit)
                
                # Convert events to serializable format
                serialized_events = []
                for event in events:
                    event_data = {
                        'event_id': event.event_id,
                        'event_type': event.event_type,
                        'timestamp': event.timestamp.isoformat(),
                        'source': event.source
                    }
                    
                    # Add event type specific data
                    if hasattr(event, 'message'):
                        event_data['message'] = event.message
                    if hasattr(event, 'data'):
                        event_data['data'] = event.data
                    if hasattr(event, 'action'):
                        event_data['action'] = event.action
                    
                    serialized_events.append(event_data)
                
                return jsonify({
                    'events': serialized_events,
                    'total': len(serialized_events)
                })
            except Exception as e:
                logger.error(f"Error getting event history: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/subscriptions')
        def get_subscriptions():
            """Get subscription info"""
            try:
                event_bus = get_global_event_bus()
                subscriptions_data = []
                
                for event_type, subscriptions in event_bus.subscriptions.items():
                    for sub in subscriptions:
                        sub_data = {
                            'subscription_id': sub.subscription_id,
                            'event_type': event_type,
                            'subscriber_id': sub.subscriber_id,
                            'priority': sub.priority,
                            'call_count': sub.call_count,
                            'error_count': sub.error_count,
                            'created_at': sub.created_at.isoformat(),
                            'last_called': sub.last_called.isoformat() if sub.last_called else None,
                            'last_error': sub.last_error
                        }
                        subscriptions_data.append(sub_data)
                
                return jsonify({
                    'subscriptions': subscriptions_data,
                    'total': len(subscriptions_data)
                })
            except Exception as e:
                logger.error(f"Error getting subscriptions: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health')
        def get_health():
            """Get health status"""
            try:
                event_bus = get_global_event_bus()
                health_status = asyncio.run(event_bus.health_check())
                return jsonify(health_status)
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/events/publish', methods=['POST'])
        def publish_test_event():
            """Publish test event"""
            try:
                data = request.get_json()
                event_type = data.get('event_type', 'system')
                message = data.get('message', 'Test event from UI')
                source = data.get('source', 'event-ui')
                
                # Create test event
                if event_type == 'system':
                    event = SystemEvent(
                        message=message,
                        source=source,
                        data=data.get('data', {})
                    )
                elif event_type == 'task':
                    event = TaskEvent(
                        task_id=data.get('task_id', 'test-task'),
                        action=data.get('action', 'test'),
                        source=source,
                        data=data.get('data', {})
                    )
                else:
                    event = SystemEvent(
                        message=message,
                        source=source,
                        data=data.get('data', {})
                    )
                
                # Publish event
                event_bus = get_global_event_bus()
                asyncio.run(event_bus.publish(event))
                
                return jsonify({
                    'success': True,
                    'event_id': event.event_id,
                    'message': 'Event published successfully'
                })
            except Exception as e:
                logger.error(f"Error publishing test event: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_events(self) -> None:
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Client connected"""
            logger.info(f"Client connected: {request.sid}")
            emit('connected', {'message': 'Connected to Event UI'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Client disconnected"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            """Start real-time monitoring"""
            logger.info("Starting real-time monitoring")
            self._start_monitoring()
            emit('monitoring_started', {'message': 'Real-time monitoring started'})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            """Stop real-time monitoring"""
            logger.info("Stopping real-time monitoring")
            self._stop_monitoring()
            emit('monitoring_stopped', {'message': 'Real-time monitoring stopped'})
    
    def _start_monitoring(self) -> None:
        """Start monitoring event bus"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._event_bus = get_global_event_bus()
        
        # Run monitoring in background thread
        def monitor_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._monitor_loop())
        
        thread = threading.Thread(target=monitor_thread, daemon=True)
        thread.start()
    
    def _stop_monitoring(self) -> None:
        """Stop monitoring"""
        self._monitoring = False
    
    async def _monitor_loop(self) -> None:
        """Monitoring loop"""
        last_stats = None
        
        while self._monitoring:
            try:
                if self._event_bus:
                    # Get current statistics
                    current_stats = self._event_bus.get_statistics()
                    
                    # If statistics changed, send update
                    if current_stats != last_stats:
                        self.socketio.emit('stats_update', current_stats)
                        last_stats = current_stats.copy()
                    
                    # Get performance metrics
                    performance = self._event_bus.get_performance_metrics()
                    self.socketio.emit('performance_update', performance)
                    
                    # Get recent events
                    recent_events = self._event_bus.get_event_history(limit=10)
                    if recent_events:
                        # Convert to serializable format
                        serialized_events = []
                        for event in recent_events:
                            event_data = {
                                'event_id': event.event_id,
                                'event_type': event.event_type,
                                'timestamp': event.timestamp.isoformat(),
                                'source': event.source
                            }
                            
                            if hasattr(event, 'message'):
                                event_data['message'] = event.message
                            if hasattr(event, 'action'):
                                event_data['action'] = event.action
                            
                            serialized_events.append(event_data)
                        
                        self.socketio.emit('events_update', {
                            'events': serialized_events
                        })
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    def run(self, debug: bool = False) -> None:
        """Run UI server
        
        Args:
            debug: Whether to enable debug mode
        """
        logger.info(f"Starting Event UI server on {self.host}:{self.port}")
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )


# Global UI server instance
_ui_server: Optional[EventUIServer] = None


def get_ui_server() -> EventUIServer:
    """Get UI server instance
    
    Returns:
        UI server instance
    """
    global _ui_server
    if _ui_server is None:
        _ui_server = EventUIServer()
    return _ui_server


def start_ui_server(host: str = '127.0.0.1', port: int = 5000, debug: bool = False) -> None:
    """Start UI server
    
    Args:
        host: Server host address
        port: Server port
        debug: Whether to enable debug mode
    """
    global _ui_server
    _ui_server = EventUIServer(host, port)
    _ui_server.run(debug)


if __name__ == '__main__':
    # Start UI server when run directly
    start_ui_server(debug=True)
