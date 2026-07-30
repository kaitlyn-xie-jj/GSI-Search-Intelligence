import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.utils.location_utils import infer_nearest_location, create_centered_shape

logger = logging.getLogger(__name__)


class ContextHub:
    """
    Context hub - receives and manages states, operations, and results from the execution layer.
    Simulates ROS-style pub/sub messaging.
    """
    
    def __init__(self, scene_graph: AbstractSceneGraph, sync_interval: float = 0.5, pos_eps: float = 0.30):
        """
        Initialize the context hub.
        
        Args:
            scene_graph: scene graph
            sync_interval: sync interval in seconds
            pos_eps: position change threshold
        """
        self.scene_graph = scene_graph
        self.motion_controller = None  # set by the execution layer
        
        # Message queues (simulating ROS topics)
        self.ops_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self.outcome_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self.state_queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        
        # Background tasks
        self._state_receiver_task: Optional[asyncio.Task] = None
        self._ops_processor_task: Optional[asyncio.Task] = None
        self._outcome_aggregator_task: Optional[asyncio.Task] = None
        
        # Parameters
        self._sync_interval = sync_interval
        self._sync_pos_eps = pos_eps
        self._force_sync_lock = asyncio.Lock()
        self._state_cycle_count = 0
        self._state_cycle_cond = asyncio.Condition()
        
        # Result store
        self._outcome_store: List[Dict] = []
        self._outcome_store_lock = asyncio.Lock()
        
        # Position cache
        self._last_positions: Dict[int, List[float]] = {}
        
        # Running flag
        self._running = False
    
    def set_motion_controller(self, motion_controller):
        """Set the motion controller."""
        self.motion_controller = motion_controller

    async def start(self):
        """Start background services."""
        if self._running:
            return
        
        self._running = True
        
        # Start receiver task
        self._state_receiver_task = asyncio.create_task(
            self._state_receiver(self._sync_interval, self._sync_pos_eps)
        )
        
        # Start ops processor task
        self._ops_processor_task = asyncio.create_task(
            self._ops_processor()
        )
        
        # Start outcome aggregator task
        self._outcome_aggregator_task = asyncio.create_task(
            self._outcome_aggregator()
        )
        
        logger.info("ContextHub services started")
    
    async def stop(self):
        """Stop background services."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all tasks
        for task in (self._state_receiver_task, self._ops_processor_task, self._outcome_aggregator_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._state_receiver_task = None
        self._ops_processor_task = None
        self._outcome_aggregator_task = None
        
        logger.info("ContextHub services stopped")
    
    # Send interfaces (used by the execution layer)
    
    async def enqueue_ops(self, operations: List[Dict], context: Dict):
        """Enqueue operations."""
        try:
            await self.ops_queue.put({
                "ops": operations,
                "context": context,
                "timestamp": datetime.now().isoformat()
            })
        except asyncio.QueueFull:
            logger.warning("Ops queue full; dropping operations batch")
    
    async def enqueue_outcomes(self, outcomes: List[Dict], meta: Dict):
        """Enqueue outcomes."""
        try:
            await self.outcome_queue.put({
                "outcomes": outcomes,
                "meta": meta,
                "timestamp": datetime.now().isoformat()
            })
        except asyncio.QueueFull:
            logger.warning("Outcome queue full; dropping outcomes")
    
    async def enqueue_state(self, entity_id: int, position: List[float]):
        """Enqueue a state update."""
        try:
            await self.state_queue.put({
                "entity_id": entity_id,
                "position": position,
                "timestamp": datetime.now().isoformat()
            })
        except asyncio.QueueFull:
            logger.debug("State queue full; dropping state update")
    
    # Sync interfaces
    
    async def drain_ops(self, timeout: float = 0.5):
        """Wait for the ops queue to drain."""
        try:
            await asyncio.wait_for(self.ops_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug("Ops drain timeout")
    
    async def drain_outcomes(self, timeout: float = 0.5):
        """Wait for the outcome queue to drain."""
        try:
            await asyncio.wait_for(self.outcome_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug("Outcome drain timeout")
    
    async def collect_outcomes(self, drain: bool = True) -> List[Dict]:
        """Collect accumulated outcomes."""
        async with self._outcome_store_lock:
            data = list(self._outcome_store)
            if drain:
                self._outcome_store.clear()
            return data
    
    async def _state_receiver_pass(self, pos_eps: float = None):
        """
        Run one full state-receive-and-apply pass (no sleep).
        - Snapshots motion_controller._entity_motion_states
        - Optionally drains externally pushed states from state_queue
        - Computes necessary node updates and awaits completion
        """
        pos_eps = self._sync_pos_eps if pos_eps is None else pos_eps

        # 1) Drain state_queue (push mode — active when callers use enqueue_state)
        drain_tasks = []
        while True:
            try:
                pkt = self.state_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                ent = int(pkt.get("entity_id"))
                pos = pkt.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    last = self._last_positions.get(ent)
                    if not last or ((pos[0]-last[0])**2 + (pos[1]-last[1])**2) ** 0.5 >= pos_eps:
                        self._last_positions[ent] = [float(pos[0]), float(pos[1])]
                        drain_tasks.append(self._update_node_shape(ent, pos))
            finally:
                # Mark done regardless of success to avoid blocking join
                self.state_queue.task_done()

        if drain_tasks:
            results = await asyncio.gather(*drain_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.debug(f"_update_node_shape (from state_queue) raised: {r}")

        # 2) Pull a snapshot from motion_controller (pull mode)
        if self.motion_controller:
            try:
                raw_states = getattr(self.motion_controller, "_entity_motion_states", {}) or {}
                state_lock = getattr(self.motion_controller, "_state_lock", None)
                if state_lock is not None:
                    try:
                        import asyncio as _asyncio
                        if isinstance(state_lock, _asyncio.Lock):
                            async with state_lock:
                                states_snapshot = dict(raw_states)
                        else:
                            from threading import Lock as _TLock
                            if isinstance(state_lock, _TLock):
                                with state_lock:
                                    states_snapshot = dict(raw_states)
                            else:
                                states_snapshot = dict(raw_states)
                    except Exception:
                        states_snapshot = dict(raw_states)
                else:
                    states_snapshot = dict(raw_states)
            except Exception:
                states_snapshot = {}

            items = list(states_snapshot.items())
            update_tasks = []
            for entity_id, position in items:
                if not position or len(position) != 2:
                    continue
                last_pos = self._last_positions.get(entity_id)
                if last_pos:
                    dx = float(position[0] - last_pos[0])
                    dy = float(position[1] - last_pos[1])
                    delta = (dx*dx + dy*dy) ** 0.5
                    if delta < pos_eps:
                        continue
                self._last_positions[entity_id] = [float(position[0]), float(position[1])]
                update_tasks.append(self._update_node_shape(entity_id, position))

            if update_tasks:
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        logger.debug(f"_update_node_shape (from mc snapshot) raised: {r}")

    async def _state_receiver(self, interval: float = 0.1, pos_eps: float = 0.3):
        try:
            while self._running:
                await self._state_receiver_pass(pos_eps)
                async with self._state_cycle_cond:
                    self._state_cycle_count += 1
                    self._state_cycle_cond.notify_all()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"State receiver error: {e}", exc_info=True)

    async def force_state_sync(self):
        """
        Run one state sync immediately at the call site, without waiting for the next tick.
        Can coexist with the background _state_receiver; uses a lock to prevent re-entry.
        """
        async with self._force_sync_lock:
            await self._state_receiver_pass()
            # Also bump the heartbeat so external waiters can detect the next sync
            async with self._state_cycle_cond:
                self._state_cycle_count += 1
                self._state_cycle_cond.notify_all()

    async def await_quiescence(self, timeout: float = 1.5, include_state: bool = True) -> bool:
        """
        Wait until the ContextHub reaches approximate quiescence:
        1) Wait for ops/outcomes queues to join
        2) Run one forced state sync to cover state changes caused by ops
        Returns True if quiescent, False if timed out (non-blocking).
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0.0, timeout)

        async def _with_remaining(coro):
            remaining = max(0.0, deadline - loop.time())
            try:
                return await asyncio.wait_for(coro, timeout=remaining)
            except asyncio.TimeoutError:
                return None

        await _with_remaining(self.drain_ops(timeout=max(0.0, deadline - loop.time())))
        await _with_remaining(self.drain_outcomes(timeout=max(0.0, deadline - loop.time())))

        if include_state:
            await _with_remaining(self.force_state_sync())

        return loop.time() <= deadline

    async def _ops_processor(self):
        """
        Ops processor - handles graph operations from the execution layer.
        Simulates a ROS ops-topic subscriber.
        """
        try:
            while self._running:
                # Receive operation from queue
                packet = await self.ops_queue.get()
                
                try:
                    ops = packet.get("ops") or []
                    context = packet.get("context") or {}
                    
                    # Execute each operation
                    for op in ops:
                        try:
                            await self._execute_operation(op, context)
                        except Exception as e:
                            logger.error(f"Failed to execute operation {op}: {e}")
                
                finally:
                    self.ops_queue.task_done()
        
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Ops processor error: {e}", exc_info=True)
    
    async def _outcome_aggregator(self):
        """
        Outcome aggregator - collects and stores execution results.
        Simulates a ROS result-topic subscriber.
        """
        try:
            while self._running:
                # Receive outcome from queue
                packet = await self.outcome_queue.get()
                
                try:
                    outcomes = packet.get("outcomes") or []
                    meta = packet.get("meta") or {}
                    
                    # Store outcomes
                    async with self._outcome_store_lock:
                        for outcome in outcomes:
                            # Attach metadata
                            outcome_meta = outcome.setdefault("meta", {})
                            for k, v in meta.items():
                                outcome_meta.setdefault(k, v)
                        
                        self._outcome_store.extend(outcomes)
                
                finally:
                    self.outcome_queue.task_done()
        
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Outcome aggregator error: {e}", exc_info=True)
    
    # Internal helpers
    
    async def _update_node_shape(self, entity_id: int, position: List[float]):
        """Update a node's shape to reflect a new position."""
        try:
            node = self.scene_graph.get_node_by_id(entity_id)
            if not node:
                return
            
            # Skip static buildings
            category = node.get("properties", {}).get("category")
            if category == "building":
                return
            
            # Build new shape
            new_shape = create_centered_shape(node, position)
            inferred_loc = infer_nearest_location(self.scene_graph, position, exclude_id=entity_id)

            payload = {"id": entity_id, "shape": new_shape}
            if inferred_loc:
                payload["location"] = inferred_loc
            
            # Update node
            await self.scene_graph.update_node_async(payload)
            
            logger.debug(f"Updated position for entity {entity_id}: {position}")
        
        except Exception as e:
            logger.debug(f"Failed to update shape for entity {entity_id}: {e}")
    
    async def _execute_operation(self, operation: Dict, context: Dict):
        """Execute a graph operation."""
        op_type = operation.get("type")
        
        if op_type == "update_node":
            target_id = self._resolve_reference(operation["target_id"], context)
            updates = operation["updates"]
            
            # Handle location update
            if "location" in updates:
                loc_id = updates["location"]
                location_node = self.scene_graph.get_node_by_id(loc_id)
                if location_node:
                    lprops = (location_node.get("properties") or {})
                    updates["location"] = {
                        "category": lprops.get("category", "unknown"),
                        "type": lprops.get("type", "Unknown"),
                        "label": lprops.get("label", "Unknown Location"),
                    }
                    # Align to target center
                    node = self.scene_graph.get_node_by_id(target_id)
                    if node:
                        updates["shape"] = self._align_shapes(node, location_node)
            
            await self.scene_graph.update_node_async({"id": target_id, **updates})
            logger.info(f"Updated node {target_id}: {updates}")
        
        elif op_type == "add_edge":
            edge_data = {
                "source": self._resolve_reference(operation["from"], context),
                "target": self._resolve_reference(operation["to"], context),
                "type": operation["edge_type"]
            }
            await self.scene_graph.add_edge_async(edge_data)
            logger.info(f"Added edge: {edge_data}")
        
        elif op_type == "remove_edge":
            edge_data = {
                "source": self._resolve_reference(operation["from"], context),
                "target": self._resolve_reference(operation["to"], context),
                "type": operation["edge_type"]
            }
            await self.scene_graph.remove_edge_async(edge_data)
            logger.info(f"Removed edge: {edge_data}")
    
    def _resolve_reference(self, reference: Any, context: Dict) -> int:
        """Resolve a reference to an actual ID."""
        if isinstance(reference, (int, float)):
            return int(reference)
        
        # Predefined reference map
        ref_map = {
            "robot": "robot_id",
            "object": "object_id",
            "carrier": "carrier_id",
            "{robot_id}": "robot_id",
            "{object_id}": "object_id",
            "{carrier_id}": "carrier_id",
        }
        
        if reference in ref_map:
            return context.get(ref_map[reference])
        
        if isinstance(reference, str) and reference.startswith("{") and reference.endswith("}"):
            key = reference[1:-1]
            return context.get(key)
        
        return reference
    
    def _align_shapes(self, source: dict, target: dict) -> dict:
        """Align the source shape to the center of the target shape."""
        target_shape = target.get('shape', {})
        target_type = target_shape.get('type')
        
        # Get target center
        if target_type == 'rectangle':
            min_corner = target_shape['min_corner']
            max_corner = target_shape['max_corner']
            center = [(min_corner[0] + max_corner[0]) / 2, (min_corner[1] + max_corner[1]) / 2]
        elif target_type == 'circle':
            center = target_shape['center']
        else:
            return source.get('shape', {})
        
        # Adjust source shape
        source_shape = source.get('shape', {})
        source_type = source_shape.get('type')
        
        if source_type == 'rectangle':
            min_corner = source_shape['min_corner']
            max_corner = source_shape['max_corner']
            width = max_corner[0] - min_corner[0]
            height = max_corner[1] - min_corner[1]
            return {
                'type': 'rectangle',
                'min_corner': [center[0] - width/2, center[1] - height/2],
                'max_corner': [center[0] + width/2, center[1] + height/2]
            }
        elif source_type == 'circle':
            return {
                'type': 'circle',
                'center': center,
                'radius': source_shape.get('radius', 1.0)
            }
        
        return source_shape
