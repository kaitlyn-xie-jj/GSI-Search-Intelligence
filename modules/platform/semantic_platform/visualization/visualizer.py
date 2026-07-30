# visualizer.py
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from matplotlib.patches import Rectangle, Circle
from typing import Dict, Any, List, Set, Tuple, Optional
import time
from itertools import chain
import matplotlib.patheffects as path_effects

from modules.config import unified_template_manager
from modules.dataset_builder.scene_utils.perlin_utils import apply_perlin_to_areas_consistent
from .utils import Views, RealtimeController


class RealTimeScenarioVisualizer:
    """
    Real-time scenario visualizer.
    - Handles assembly: loads scene, creates Figure/Axes, color mapping.
    - Static rendering delegated to Views (views.py).
    - Dynamic elements and animation loop delegated to RealtimeController (realtime.py).
    """

    def __init__(self,
                 title: str = "Real-Time Scenario Visualization",
                 enable_motion_trails: bool = True,
                 motion_update_rate: int = 20,
                 goal: Optional[Dict[str, Any]] = None,
                 video_writer: Optional[Any] = None,
                 noise_post: bool = True,
                 noise_amp: float = 20.0,
                 noise_scale: float = 0.01,
                 noise_octaves: int = 1,
                 noise_maxseg: float = 8.0,
                 seed: Optional[int] = 42,
                 show_relation_graph: bool = False):
        """
        Args:
            title: window title
            enable_motion_trails: whether to display motion trails
            motion_update_rate: motion update frequency (Hz), recorded here but animation managed by RealtimeController
            video_writer: optional video recorder
            noise_post: whether to apply Perlin post-processing to area boundaries (visual only)
            noise_amp: Perlin amplitude (meters)
            noise_scale: Perlin scale
            noise_octaves: Perlin octaves
            noise_maxseg: edge subdivision length before jitter (meters)
            seed: random seed
            show_relation_graph: whether to display semantic relation graph
        """
        self.show_relation_graph = show_relation_graph

        # ===== Load scene =====
        from modules.utils import get_project_root
        output_dir = get_project_root() / "dataset" / "scenarios" / "cybertown" / "scenario_1"
        filepath = output_dir / "scene_graph.json"
        with open(filepath, 'r', encoding='utf-8') as f:
            scene = json.load(f)

        self.nodes = scene["nodes"]
        self.edges = scene["edges"]
        self.bounds = {'x_min': 0, 'x_max': 1000, 'y_min': 0, 'y_max': 1000}
        self.previous_nodes: List[Dict[str, Any]] = []
        self.previous_edges: List[Dict[str, Any]] = []
        self.goal = goal

        # Apply Perlin noise post-processing (visual decoration only)
        self.perlin_info = None
        if noise_post:
            try:
                self.perlin_info = apply_perlin_to_areas_consistent(
                    self.nodes,
                    types=("water_body", "garden"),   # can extend types as needed
                    amplitude=noise_amp,
                    scale=noise_scale,
                    octaves=noise_octaves,
                    max_seg_len=noise_maxseg,
                )
            except Exception as e:
                print(f"Warning: Perlin noise application failed: {e}")

        self.nodes_for_view = self.perlin_info.get("nodes", self.nodes)

        # ===== Matplotlib basic setup =====
        plt.ion()
        plt.style.use('default')
        if self.show_relation_graph:
            # Keep original dual-view layout
            self.fig = plt.figure(figsize=(24, 12))
            self.fig.patch.set_facecolor('white')
            self.ax_spatial = self.fig.add_axes([0.01, 0.02, 0.61, 0.96])  # left: spatial
            self.ax_relation = self.fig.add_axes([0.62, 0.2, 0.38, 0.78])  # right: relation
        else:
            # Single-view layout, canvas nearly square
            self.fig = plt.figure(figsize=(12, 12))
            self.fig.patch.set_facecolor('white')
            self.ax_spatial = self.fig.add_axes([0.02, 0.02, 0.96, 0.96])   # centered
            self.ax_relation = None  # relation graph axes not present

        # Try to set window geometry (compatible with different backends)
        try:
            manager = plt.get_current_fig_manager()
            if self.show_relation_graph:
                width, height, x_pos, y_pos = 2000, 1300, 1500, 100
            else:
                width, height, x_pos, y_pos = 1300, 1300, 1500, 100
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'geometry'):
                    manager.window.geometry(f"{width}x{height}+{x_pos}+{y_pos}")
                elif hasattr(manager.window, 'setGeometry'):
                    manager.window.setGeometry(x_pos, y_pos, width, height)
            elif hasattr(manager, 'frame'):
                if hasattr(manager.frame, 'SetSize'):
                    manager.frame.SetSize((width, height))
                    manager.frame.SetPosition((x_pos, y_pos))
            else:
                self.fig.set_size_inches(width / 100, height / 100)
        except Exception:
            try:
                self.fig.set_size_inches(24, 12)
            except Exception:
                pass

        # ===== Color mapping =====
        self.color_map = self._generate_dynamic_color_map()
        rels = sorted({e.get("type") for e in self.edges if e.get("type")})
        edge_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
        self.rel_color_map = {r: edge_colors[i % len(edge_colors)] for i, r in enumerate(rels)}

        # ===== Assemble views and realtime controller =====
        self.views = Views(self.fig, self.ax_spatial, self.ax_relation,
                           self.color_map, self.rel_color_map, self.bounds)

        self.realtime = RealtimeController(
            fig=self.fig,
            ax_spatial=self.ax_spatial,
            ax_relation=self.ax_relation,
            color_map=self.color_map,
            views=self.views,
            video_writer=video_writer,
            enable_motion_trails=enable_motion_trails,
            max_trail_length=10,
            inactive_timeout=1.5
        )

        # Set initial scene for realtime layer
        self.realtime.set_scene(self.nodes, self.edges)

        # ===== First frame rendering =====
        self.update(self.nodes, self.edges, timestep=0)

    # ===================== Public API =====================

    def update(self, new_nodes: List[Dict[str, Any]], new_edges: List[Dict[str, Any]], timestep: int):
        """
        Update and redraw with new scene state (static layer + relation graph)
        Dynamic element updates driven by RealtimeController animation frames.
        """
        # Record to realtime layer for node/edge lookup
        self.realtime.set_scene(new_nodes, new_edges)

        # Calculate edge additions/deletions (for relation graph highlight titles)
        added_edges, deleted_edges = [], []
        if self.previous_edges:
            prev_edge_set = {self._get_edge_identifier(e) for e in self.previous_edges}
            current_edge_set = {self._get_edge_identifier(e) for e in new_edges}
            added_edge_ids = current_edge_set - prev_edge_set
            deleted_edge_ids = prev_edge_set - current_edge_set
            added_edges = [e for e in new_edges if self._get_edge_identifier(e) in added_edge_ids]
            deleted_edges = [e for e in self.previous_edges if self._get_edge_identifier(e) in deleted_edge_ids]

        # Save old dynamic objects, clear static canvas
        self._clear_static_content()

        # Static spatial map
        carried_ids = {e['target'] for e in new_edges if e.get('type') == 'carrying'}
        self.views.draw_spatial(self.nodes_for_view, new_edges,
                                moving_entities=self.realtime.moving_entities,
                                carried_prop_ids=carried_ids,
                                perlin_edge_curves=self.perlin_info["edge_curves"] if self.perlin_info else None,
                                perlin_edge_curves_loose=self.perlin_info["edge_curves_loose"] if self.perlin_info else None,
                                street_match_quant=self.perlin_info["street_match_quant"] if self.perlin_info else 0.5,)
        
        # Draw search area for current task
        if getattr(self, "goal", None):
            try:
                self.views.draw_goal_area(self.goal, edge_color="#D27219", fill_alpha=0.08, line_width=2.4)
            except Exception as e:
                print(f"[draw_goal_area] warn: {e}")

        # Relation graph
        if self.show_relation_graph:
            self.views.draw_graph(new_nodes, new_edges, added_edges, deleted_edges)

        # Ensure dynamic elements on top
        self._refresh_dynamic_elements()

        # Refresh canvas
        self.fig.canvas.draw_idle()

        # Persist current state
        self.nodes, self.edges = new_nodes, new_edges
        self.previous_nodes = deepcopy(new_nodes)
        self.previous_edges = deepcopy(new_edges)

    def update_entity_position(self, entity_id: int, position: List[float], additional_info: Optional[Dict] = None):
        """
        Push realtime position into queue (consumed by RealtimeController in animation frames).
        """
        self.realtime.enqueue_position(entity_id, position, additional_info)

    def update_edge_state(self, source_id: int, target_id: int, edge_type: str, action: str = 'highlight'):
        """
        Realtime semantic edge highlight: 'highlight' / 'normal'
        """
        self.realtime.enqueue_edge_highlight(source_id, target_id, edge_type, action)

    def set_video_writer(self, video_writer: Any):
        """Optional: set/replace video recorder"""
        # RealtimeController will call it internally during redraw
        self.realtime.loop.video_writer = video_writer

    def stop_animation(self):
        """Explicitly stop background animation loop."""
        try:
            if self.realtime and self.realtime.loop and self.realtime.loop.animation:
                self.realtime.loop.animation.event_source.stop()
        except Exception as e:
            pass 
    def close(self):
        """Close window"""
        self.stop_animation()
        plt.ioff()
        plt.close(self.fig)
        print("Visualization window closed.")

    # ===================== Internal utilities =====================

    def _get_edge_identifier(self, edge: Dict) -> Tuple:
        """Create unique hashable identifier for edge"""
        return (edge['source'], edge['target'], edge.get('type', ''))

    def _generate_dynamic_color_map(self) -> Dict[str, str]:
        # Use unified_template_manager to list all types
        types: List[str] = []
        for cat in unified_template_manager.categories():
            types.extend(unified_template_manager.get_all_types(cat))
        types = sorted(set(types))

        # Soft color palette (consistent with original)
        cmaps = [plt.cm.Set3.colors, plt.cm.Pastel1.colors, plt.cm.Pastel2.colors]
        palette = list(chain.from_iterable(cmaps))

        color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}
        # Manually specify fixed light colors
        color_map["UAV"] = "#a6cee3"       # light blue
        color_map["UGV"] = "#b2df8a"       # light green
        color_map["Humanoid"] = "#fdbf6f"  # light orange
        color_map["Quadruped"] = "#cab2d6" # light purple

        # -- Basic common sense fixed colors (overrides) --
        # Areas
        color_map["water_body"]        = "#a0d8ef"  # light water blue
        color_map["garden"]              = "#b8e986"  # light grass green
        color_map["greenbelt"]         = "#c7f1a3"  # lighter greenbelt
        color_map["square"]            = "#d9d9d9"  # light gray square
        color_map["campus"]            = "#b3c7ff"  # campus blue
        color_map["industrial_park"]   = "#ffcc99"  # industrial park light orange
        color_map["neighborhood"]  = "#bcdffb"  # residential area light blue

        # Infrastructure
        color_map["street_segment"]    = "#616161"  # darker road
        color_map["bridge"]            = "#455A64"  # bridge (dark gray blue)
        color_map["sidewalk"]          = "#bdbdbd"  # sidewalk light gray
        color_map["intersection"]      = "#757575"  # intersection medium gray

        # Buildings (as soft as possible)
        color_map["hospital"]          = "#ffd1dc"  # light pink (easy to identify)
        color_map["mall"]              = "#ffd59e"  # light apricot
        color_map["parking"]           = "#cfd8dc"  # cool gray
        color_map["power_station"]     = "#ffe599"  # light yellow
        color_map["library"]         = "#c5e1a5"  # mild green
        color_map["robot_base"]        = "#b0bec5"  # cool gray blue

        return color_map

    def _clear_static_content(self):
        """
        Clear static content but keep dynamic elements (handled by RealtimeController.DynamicLayer).
        """
        # Get reference to dynamic layer
        dyn = self.realtime.dyn

        # Save still valid visual elements
        saved_patches = {eid: p for eid, p in dyn.dynamic_patches.items() if getattr(p, 'figure', None) is not None}
        saved_lines = {eid: l for eid, l in dyn.trail_lines.items() if getattr(l, 'figure', None) is not None}
        saved_texts = {eid: t for eid, t in dyn.motion_info_texts.items()
                       if getattr(t, 'figure', None) is not None and getattr(t, 'axes', None) is not None}

        saved_cross_lines = {}
        if dyn.drone_cross_lines:
            for eid, (h, v) in dyn.drone_cross_lines.items():
                if getattr(h, 'figure', None) is not None and getattr(v, 'figure', None) is not None:
                    saved_cross_lines[eid] = (h, v)

        # Clear both Axes
        self.ax_spatial.cla()
        if self.ax_relation:
            self.ax_relation.cla()

        # Restore dynamic elements to new axes (re-add)
        dyn.dynamic_patches = saved_patches
        dyn.trail_lines = saved_lines
        dyn.motion_info_texts = saved_texts
        dyn.drone_cross_lines = saved_cross_lines

        for patch in dyn.dynamic_patches.values():
            self.ax_spatial.add_patch(patch)

        # Trail lines need redrawing (get their data)
        for eid, line in list(dyn.trail_lines.items()):
            xs, ys = line.get_data()
            new_line, = self.ax_spatial.plot(xs, ys, 'b-', alpha=0.3, linewidth=1)
            dyn.trail_lines[eid] = new_line

        # Drone cross lines also need rebuilding
        if dyn.drone_cross_lines:
            new_cross = {}
            for eid, (hline, vline) in dyn.drone_cross_lines.items():
                hx, hy = hline.get_data()
                vx, vy = vline.get_data()
                new_h, = self.ax_spatial.plot(hx, hy, 'darkblue', linewidth=2, zorder=11)
                new_v, = self.ax_spatial.plot(vx, vy, 'darkblue', linewidth=2, zorder=11)
                new_cross[eid] = (new_h, new_v)
            dyn.drone_cross_lines = new_cross

        # Rebuild text objects (copy styles)
        new_texts = {}
        for eid, old_text in dyn.motion_info_texts.items():
            try:
                pos = old_text.get_position()
                content = old_text.get_text()
                ha = old_text.get_ha()
                va = old_text.get_va()
                fontsize = old_text.get_fontsize()
                weight = getattr(old_text, 'get_weight', lambda: 'normal')()
                bbox_patch = old_text.get_bbox_patch()
                if bbox_patch:
                    facecolor = bbox_patch.get_facecolor()
                    alpha = bbox_patch.get_alpha()
                else:
                    facecolor, alpha = 'yellow', 0.85

                new_text = self.ax_spatial.text(
                    pos[0], pos[1], content,
                    ha=ha, va=va,
                    fontsize=fontsize,
                    weight=weight,
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor=facecolor,
                              alpha=alpha),
                    zorder=15
                )
                new_texts[eid] = new_text
            except Exception:
                pass
        dyn.motion_info_texts = new_texts

    def _refresh_dynamic_elements(self):
        """
        Ensure dynamic elements are at upper zorder
        """
        dyn = self.realtime.dyn
        for patch in dyn.dynamic_patches.values():
            patch.set_zorder(10)
        for text in dyn.motion_info_texts.values():
            text.set_zorder(15)