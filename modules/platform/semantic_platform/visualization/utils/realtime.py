# realtime.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import deque
from random import Random
import time
import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle

label_fontsize = 12

class MotionLoop:
    """
    Handles: starting animation + controlling unified redraw timing
    """
    def __init__(self, fig: plt.Figure, ax_spatial: plt.Axes, ax_relation: plt.Axes,
                 video_writer=None, force_redraw_interval: float = 0.05):
        self.fig = fig
        self.ax_spatial = ax_spatial
        self.ax_relation = ax_relation
        self.video_writer = video_writer
        self.force_redraw_interval = force_redraw_interval
        self._last = time.time()
        self.animation = None

    def start(self, on_frame_callback):
        self.animation = animation.FuncAnimation(
            self.fig,
            on_frame_callback,
            interval=25,           # ~40Hz
            blit=False,
            repeat=True,
            cache_frame_data=False
        )
        return self.animation

    def maybe_redraw(self):
        now = time.time()
        if now - self._last > self.force_redraw_interval:
            self.ax_spatial.figure.canvas.draw_idle()
            if self.ax_relation:
                self.ax_relation.figure.canvas.draw_idle()
            self.ax_spatial.figure.canvas.flush_events()
            self._last = now
            if self.video_writer:
                self.video_writer.update()


class DynamicLayer:
    """
    Dynamic entity layer: creates and updates patches (shapes) / trail lines / info texts.
    Kept separate from the static base map to avoid redundant redrawing.
    """
    def __init__(self, ax_spatial: plt.Axes, color_map: Dict[str, str]):
        self.ax = ax_spatial
        self.color_map = color_map

        # Visual object caches
        self.dynamic_patches: Dict[int, Any] = {}      # entity_id -> patch
        self.trail_lines: Dict[int, Any] = {}          # entity_id -> Line2D
        self.motion_info_texts: Dict[int, Any] = {}    # entity_id -> Text
        self.entity_original_colors: Dict[int, str] = {}  # entity_id -> color

        # UAV crosshair lines
        self.drone_cross_lines: Dict[int, Tuple[Any, Any]] = {}

    # --------------- Patches ---------------
    def create_patch(self, entity_id: int, node: Dict[str, Any], position: List[float]) -> None:
        props = node['properties']
        entity_type = props.get('type', props.get('category', ''))
        category = props.get('category')

        color = self.color_map.get(entity_type, '#C0C0C0')
        self.entity_original_colors[entity_id] = color

        if category == 'robot':
            if entity_type == 'UAV':
                circle = Circle(position, radius=3,
                                facecolor=color, edgecolor='darkblue',
                                linewidth=2, alpha=0.8, zorder=16)
                self.ax.add_patch(circle)
                self.dynamic_patches[entity_id] = circle

                # UAV crosshair
                cross_size = 2
                hline, = self.ax.plot([position[0] - cross_size, position[0] + cross_size],
                                      [position[1], position[1]], 'darkblue', linewidth=2, zorder=11)
                vline, = self.ax.plot([position[0], position[0]],
                                      [position[1] - cross_size, position[1] + cross_size], 'darkblue',
                                      linewidth=2, zorder=11)
                self.drone_cross_lines[entity_id] = (hline, vline)
            else:
                rect = Rectangle((position[0] - 2, position[1] - 2), 4, 4,
                                 facecolor=color, edgecolor='darkgreen',
                                 linewidth=2, alpha=0.8, zorder=16)
                self.ax.add_patch(rect)
                self.dynamic_patches[entity_id] = rect

            # Robots get a trail line by default
            line, = self.ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
            self.trail_lines[entity_id] = line

        elif category == 'prop':
            # Vehicles: rectangle; others: circle
            if 'vehicle' in entity_type.lower() or 'vehicle' in entity_type.lower():
                rect = Rectangle((position[0] - 2.5, position[1] - 1.5), 5, 3,
                                 facecolor=color, edgecolor='red',
                                 linewidth=2, alpha=0.9, zorder=10)
                self.ax.add_patch(rect)
                self.dynamic_patches[entity_id] = rect
            else:
                circle = Circle(position, radius=1.5,
                                facecolor=color, edgecolor='red',
                                linewidth=2, alpha=0.9, zorder=9)
                self.ax.add_patch(circle)
                self.dynamic_patches[entity_id] = circle

    def update_patch(self, entity_id: int, position: List[float]) -> None:
        patch = self.dynamic_patches.get(entity_id)
        if not patch:
            return
        if isinstance(patch, Circle):
            patch.center = position
            if entity_id in self.drone_cross_lines:
                hline, vline = self.drone_cross_lines[entity_id]
                cross_size = 2
                hline.set_data([position[0] - cross_size, position[0] + cross_size],
                               [position[1], position[1]])
                vline.set_data([position[0], position[0]],
                               [position[1] - cross_size, position[1] + cross_size])
        elif isinstance(patch, Rectangle):
            # Both robot and prop rectangles are aligned by center.
            # Robot is 4x4; vehicle is 5x3 (here the robot 4x4 center offset is used;
            # where a vehicle exists, it was already initialized with 5x3).
            w = patch.get_width()
            h = patch.get_height()
            patch.set_xy((position[0] - w/2, position[1] - h/2))

    # --------------- Trail lines ---------------
    def update_trail(self, entity_id: int, points: List[List[float]]) -> None:
        line = self.trail_lines.get(entity_id)
        if not line:
            return
        if len(points) > 1:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            line.set_data(xs, ys)
        else:
            line.set_data([], [])

    # --------------- Info text box (robot) ---------------
    def update_motion_info(self, entity_id: int, position: List[float], info: Dict[str, Any],
                           nodes: List[Dict[str, Any]]) -> None:
        if 'skill' not in info:
            # Hide when there is no skill
            if entity_id in self.motion_info_texts:
                self.remove_text(entity_id)
            return

        # Get robot color
        robot_color = None
        for n in nodes:
            if n['id'] == entity_id:
                robot_type = n['properties'].get('type', n['properties'].get('category', ''))
                robot_color = self.color_map.get(robot_type, 'lightgreen')
                break
        robot_color = robot_color or 'lightgreen'

        # Text content
        parts = []
        label = info.get('label', '')
        if not label:
            for n in nodes:
                if n['id'] == entity_id:
                    label = n['properties'].get('label', f'Entity_{entity_id}')
                    break
        if label:
            parts.append(f"{label}")
        skill_line = f"{info['skill']}"
        if 'phase' in info:
            skill_line += f" - {info['phase']}"
        parts.append(skill_line)
        if 'progress' in info:
            parts.append(f"Progress: {info['progress']}%")
        if 'altitude' in info:
            parts.append(f"Alt: {info['altitude']}m")
        text_content = "\n".join(parts)

        bbox_props = dict(boxstyle='round,pad=0.5', alpha=0.85, edgecolor='black', linewidth=1,
                          facecolor=robot_color)

        riding_on_id = info.get('riding_on')
        text_offset_y = 70 if riding_on_id else 8

        if entity_id not in self.motion_info_texts:
            text = self.ax.text(
                position[0], position[1] + text_offset_y, text_content,
                ha='center', va='bottom', fontsize=label_fontsize, weight='bold',
                bbox=bbox_props, zorder=50
            )
            self.motion_info_texts[entity_id] = text
        else:
            text = self.motion_info_texts[entity_id]
            text.set_position((position[0], position[1] + text_offset_y))
            text.set_text(text_content)
            text.set_bbox(bbox_props)
            text.set_visible(True)
            text.set_zorder(50)

    # --------------- Info text box (prop) ---------------
    def update_carried_prop_info(self, entity_id: int, position: List[float], info: Dict[str, Any],
                                 nodes: List[Dict[str, Any]], color_fallback: str = "#C0C0C0") -> None:
        label = info.get('label', '')
        prop_color = self.entity_original_colors.get(entity_id)

        if not prop_color:
            for n in nodes:
                if n['id'] == entity_id:
                    if not label:
                        label = n['properties'].get('label', f'Prop_{entity_id}')
                    prop_type = n['properties'].get('type', n['properties'].get('category', ''))
                    prop_color = self.color_map.get(prop_type, color_fallback)
                    self.entity_original_colors[entity_id] = prop_color
                    break
        if not prop_color:
            prop_color = color_fallback

        is_moving = info.get('status') == 'MOVING' or info.get('motion_type') == 'autonomous'
        should_display = info.get('carried') or info.get('is_target') or is_moving or label

        if not should_display:
            if entity_id in self.motion_info_texts:
                self.remove_text(entity_id)
            return

        # Display for different states
        if info.get('carried'):
            text_content = f"[{label}]\n(Carried)"
            bbox_props = dict(boxstyle='round,pad=0.3', alpha=0.8,
                              facecolor=prop_color, edgecolor='black', linewidth=1.5)
            text_offset_y = -5
        elif is_moving:
            text_content = f"[{label}]"
            if 'progress' in info:
                text_content += f"\n{info['progress']}"
            bbox_props = dict(boxstyle='round,pad=0.3', alpha=0.9,
                              facecolor=prop_color, edgecolor='red', linewidth=2)
            text_offset_y = 5
        else:
            if not label:
                if entity_id in self.motion_info_texts:
                    self.motion_info_texts[entity_id].set_visible(False)
                return
            text_content = f"[{label}]"
            bbox_props = dict(boxstyle='round,pad=0.3', alpha=0.7,
                              facecolor=prop_color, edgecolor='black', linewidth=1)
            text_offset_y = 3

        if entity_id not in self.motion_info_texts:
            text = self.ax.text(
                position[0], position[1] + text_offset_y, text_content,
                ha='center', va='bottom' if text_offset_y > 0 else 'top',
                fontsize=label_fontsize,
                weight='bold' if (info.get('is_target') or is_moving) else 'normal',
                bbox=bbox_props, zorder=14
            )
            self.motion_info_texts[entity_id] = text
        else:
            text = self.motion_info_texts[entity_id]
            text.set_position((position[0], position[1] + text_offset_y))
            text.set_text(text_content)
            text.set_bbox(bbox_props)
            text.set_va('bottom' if text_offset_y > 0 else 'top')
            text.set_weight('bold' if (info.get('is_target') or is_moving) else 'normal')
            text.set_visible(True)

    # --------------- Info text box (building - navigation target) ---------------
    def update_building_target_info(self, entity_id: int, position: List[float], info: Dict[str, Any]) -> None:
        if not info.get('is_navigation_target'):
            if entity_id in self.motion_info_texts:
                self.motion_info_texts[entity_id].set_visible(False)
            return

        label = info.get('label', '')
        text_content = f"{label}"
        bbox_props = dict(boxstyle='round,pad=0.4', alpha=0.8,
                          facecolor='pink', edgecolor='red', linewidth=2)

        text_offset_y = 10
        if entity_id not in self.motion_info_texts:
            text = self.ax.text(
                position[0], position[1] + text_offset_y, text_content,
                ha='center', va='bottom', fontsize=label_fontsize, weight='bold',
                bbox=bbox_props, zorder=15
            )
            self.motion_info_texts[entity_id] = text
        else:
            text = self.motion_info_texts[entity_id]
            text.set_position((position[0], position[1] + text_offset_y))
            text.set_text(text_content)
            text.set_bbox(bbox_props)
            text.set_visible(True)

    def remove_text(self, entity_id: int) -> None:
        if entity_id in self.motion_info_texts:
            t = self.motion_info_texts[entity_id]
            t.set_visible(False)
            t.remove()
            del self.motion_info_texts[entity_id]


class RealtimeController:
    """
    Realtime coordinator:
    - Consumes position/edge update queues
    - Calls DynamicLayer to update the dynamic display
    - Calls Views to highlight/unhighlight semantic edges
    - Cleans up timed-out elements
    """
    def __init__(self,
                 fig: plt.Figure,
                 ax_spatial: plt.Axes,
                 ax_relation: Optional[plt.Axes],
                 color_map: Dict[str, str],
                 views,                        # Views instance (duck-typed here to avoid circular import)
                 video_writer=None,
                 enable_motion_trails: bool = True,
                 max_trail_length: int = 10,
                 inactive_timeout: float = 1.5):
        self.enable_motion_trails = enable_motion_trails
        self.inactive_timeout = inactive_timeout
        self.max_trail_length = max_trail_length

        # Dynamic layer and animation
        self.ax_spatial = ax_spatial
        self.ax_relation = ax_relation
        self.dyn = DynamicLayer(ax_spatial, color_map)
        self.loop = MotionLoop(fig, ax_spatial, ax_relation, video_writer=video_writer)

        # Interaction with the right-side relation graph
        self.views = views

        # Realtime state
        self.real_time_positions: Dict[int, List[float]] = {}
        self.motion_trails: Dict[int, deque] = {}              # entity_id -> deque of [x,y]
        self.active_entities: Set[int] = set()
        self.entity_activity_timeout: Dict[int, float] = {}
        self.moving_entities: Set[int] = set()                 # readable from outside

        # Queues
        self.motion_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.edge_update_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        # Data references (injected by the upper layer)
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

        # Bind animation callback
        self.animation = self.loop.start(self._on_animation_frame)

    # == Public interface ==
    def set_scene(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.nodes = nodes
        self.edges = edges

    def enqueue_position(self, entity_id: int, position: List[float], additional_info: Optional[Dict[str, Any]] = None) -> None:
        self.motion_queue.put({
            'entity_id': entity_id,
            'position': position,
            'info': additional_info or {}
        })

    def enqueue_edge_highlight(self, source_id: int, target_id: int, edge_type: str, action: str = 'highlight') -> None:
        self.edge_update_queue.put({
            'edge_id': (source_id, target_id, edge_type),
            'action': action
        })

    # == Animation frame callback ==
    def _on_animation_frame(self, frame):
        # Merge the latest position updates for the same entity
        latest_positions: Dict[int, Dict[str, Any]] = {}
        while not self.motion_queue.empty():
            try:
                upd = self.motion_queue.get_nowait()
                latest_positions[upd['entity_id']] = upd
            except queue.Empty:
                break
        updates = list(latest_positions.values())

        # Process edge highlight queue
        edge_updates = []
        while not self.edge_update_queue.empty():
            try:
                e = self.edge_update_queue.get_nowait()
                edge_updates.append(e)
            except queue.Empty:
                break

        if updates:
            self._process_motion_updates(updates)
        if edge_updates:
            self._process_edge_updates(edge_updates)

        # Unified redraw and recording
        self.loop.maybe_redraw()
        return []

    # == Internal: process edge highlights ==
    def _process_edge_updates(self, updates: List[Dict[str, Any]]) -> None:
        for u in updates:
            edge_id = u['edge_id']              # (source, target, type)
            action = u.get('action', 'highlight')
            self.views.set_edge_highlight(edge_id, on=(action == 'highlight'))

    # == Internal: process motion updates ==
    def _process_motion_updates(self, updates: List[Dict[str, Any]]) -> None:
        current_time = time.time()
        updated_entities: Set[int] = set()

        # Fast index
        nodes_by_id: Dict[int, Dict[str, Any]] = {n['id']: n for n in self.nodes}

        for upd in updates:
            eid = upd['entity_id']
            pos = upd['position']
            info = upd.get('info', {})

            # Realtime position and activity timestamp
            self.real_time_positions[eid] = pos
            self.entity_activity_timeout[eid] = current_time

            # Whether active
            if 'skill' in info or info.get('is_target') or info.get('carried'):
                self.active_entities.add(eid)
            elif eid in self.active_entities and 'skill' not in info:
                self.active_entities.discard(eid)

            # Initialize patch
            if eid not in self.dyn.dynamic_patches:
                node = nodes_by_id.get(eid)
                if node:
                    self.dyn.create_patch(eid, node, pos)

            # Update position
            self.dyn.update_patch(eid, pos)

            # Trail
            if self.enable_motion_trails:
                if eid not in self.motion_trails:
                    self.motion_trails[eid] = deque(maxlen=self.max_trail_length)
                trail = self.motion_trails[eid]
                if (not trail) or (np.linalg.norm(np.array(pos) - np.array(trail[-1])) > 2.0):
                    trail.append(pos)

                # Show trail when a robot is executing a skill; otherwise clear the line
                node = nodes_by_id.get(eid)
                if node and node['properties'].get('category') == 'robot' and 'skill' in info:
                    self.dyn.update_trail(eid, list(trail))
                else:
                    if eid in self.dyn.trail_lines:
                        self.dyn.trail_lines[eid].set_data([], [])

            # Info text box
            node = nodes_by_id.get(eid)
            if node:
                cat = node['properties'].get('category')
                if cat == 'robot':
                    self.dyn.update_motion_info(eid, pos, info, self.nodes)
                elif cat == 'prop':
                    self.dyn.update_carried_prop_info(eid, pos, info, self.nodes)
                elif cat == 'building' and info.get('is_navigation_target'):
                    self.dyn.update_building_target_info(eid, pos, info)

            # Moving-state set
            is_moving = info.get('status') == 'MOVING' or info.get('motion_type') == 'autonomous' or ('skill' in info)
            if is_moving:
                self.moving_entities.add(eid)
            else:
                self.moving_entities.discard(eid)

            updated_entities.add(eid)

        # Timeout cleanup
        self._cleanup_inactive_texts(current_time)

        # Partial redraw (optional: performance optimization)
        if updated_entities and hasattr(self.ax_spatial, 'draw_artist'):
            for eid in updated_entities:
                if eid in self.dyn.dynamic_patches:
                    self.ax_spatial.draw_artist(self.dyn.dynamic_patches[eid])
                if eid in self.dyn.trail_lines:
                    self.ax_spatial.draw_artist(self.dyn.trail_lines[eid])
                if eid in self.dyn.motion_info_texts:
                    self.ax_spatial.draw_artist(self.dyn.motion_info_texts[eid])

    # == Internal: clean up inactive texts/trails ==
    def _cleanup_inactive_texts(self, current_time: float) -> None:
        to_cleanup: List[int] = []
        for eid, last in list(self.entity_activity_timeout.items()):
            if current_time - last > self.inactive_timeout and eid not in self.active_entities:
                to_cleanup.append(eid)

        for eid in to_cleanup:
            # Text
            self.dyn.remove_text(eid)
            self.entity_activity_timeout.pop(eid, None)
            # Moving state
            self.moving_entities.discard(eid)
            # Trail
            if eid in self.motion_trails:
                del self.motion_trails[eid]
            if eid in self.dyn.trail_lines:
                self.dyn.trail_lines[eid].set_data([], [])
