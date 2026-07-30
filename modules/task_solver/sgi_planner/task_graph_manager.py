
import re
import json
from copy import deepcopy 
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
import logging
from modules.utils.system.logging_utils import dlog
from modules.task_solver.sgi_planner.utils.compact_parsers import parse_compact_skill, parse_compact_edge

def canonicalize(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t_norm = str(t).strip().lower()
    aliases = {
        "uav": "UAV",
        "fw_uav": "FW_UAV",
        "ugv": "UGV",
        "quadruped": "Quadruped",
        "quad": "Quadruped",
        "humanoid": "Humanoid",
    }
    return aliases.get(t_norm, t_norm.upper())

class AtomicTaskUpdater:
    """
    Atomic task updater.
    """
    def __init__(self, logger=None, context: Optional[Any] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.context = context  

    # ===== Public Entry =====
    def update_tasks_for_event(self, tasks_list: List[Dict], event_type: str, details: Dict):
        handlers = {
            "ROBOT_FAULT": self._upd_robot_limit_by_availability,
            "ROBOT_BATTERY_LOW": self._upd_robot_limit_by_availability,
            "ROBOT_COMM_JAMMED": self._upd_robot_comm_jammed,
            "TARGET_NOT_FOUND": self._noop,
            "TARGET_UNREACHABLE": self._noop,
            "TARGET_STATUS_INCOMPATIBLE": self._noop,
            "TARGET_TYPE_INCOMPATIBLE": self._noop,
            "CARRIER_NOT_FOUND": self._noop,
            "CARRIER_LOCATION_MISMATCH": self._noop,
            "CRITICAL_PATH_BROKEN": self._noop,
            "SKILL_NOT_FOUND": self._noop,
            "ROBOT_NOT_APPLICABLE": self._noop,
            "NOOP": self._noop,
        }
        (handlers.get(event_type) or self._noop)(tasks_list, details)

    # ===== Utility Functions =====
    def _resolve_robot_type(self, details: Dict) -> Optional[str]:
        rtype = details.get("robot_type") or (details.get("robot") or {}).get("type")
        return canonicalize(rtype) if rtype else None

    def _get_available_num(self, robot_type: Optional[str]) -> Optional[int]:
        if not robot_type or not self.context:
            return None
        try:
            gt = getattr(self.context, "_generated_text", {}) or {}
            avail = gt.get("available_robots") or {}
            bucket = avail.get(robot_type) or {}
            num = bucket.get("num")
            return int(num) if num is not None else None
        except Exception:
            return None

    def _match_robot_type(self, sk, target_type: str) -> bool:
        if isinstance(sk, str):
            sk = parse_compact_skill(sk)
        rtype = sk.get("assigned_robot_type")
        if rtype is None:
            return False
        if isinstance(rtype, str):
            return rtype == target_type
        if isinstance(rtype, list):
            return target_type in rtype
        return False

    # ===== Robot Events =====
    def _upd_robot_limit_by_availability(self, tasks_list: List[Dict], details: Dict):
        """
        For skills matching robot_type, reduce assigned_robot_count by 1 when it exceeds available_num.
        The value does not go below 0. Supports compact strings and dictionary format.
        """
        rtype = self._resolve_robot_type(details)
        rlabel = (details.get("robot") or {}).get("label") or details.get("robot_label")
        available_num = self._get_available_num(rtype)

        if rtype is None or available_num is None:
            return

        changed = 0
        for task in tasks_list:
            rs = task.get("required_skills") or []
            for i, sk in enumerate(rs):
                if isinstance(sk, str):
                    parsed = parse_compact_skill(sk)
                    if not self._match_robot_type(parsed, rtype):
                        continue
                    cnt = parsed.get("assigned_robot_count", 1)
                    if isinstance(cnt, int) and cnt > available_num and cnt > 0:
                        parsed["assigned_robot_count"] = cnt - 1
                        # Write back as a compact string.
                        rt_str = "|".join(parsed["assigned_robot_type"])
                        rs[i] = f"{rt_str}:{parsed['skill_name']}:{parsed['assigned_robot_count']}"
                        changed += 1
                else:
                    if not self._match_robot_type(sk, rtype):
                        continue
                    cnt = sk.get("assigned_robot_count", 1)
                    if isinstance(cnt, int) and cnt > available_num and cnt > 0:
                        sk["assigned_robot_count"] = cnt - 1
                        changed += 1

    def _upd_robot_comm_jammed(self, tasks_list: List[Dict], details: Dict):
        """Communication is jammed; do not modify atomic tasks."""
        pass

    # ===== Placeholder =====
    def _noop(self, tasks_list: List[Dict], details: Dict):
        pass

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskGraphManager:
    """
    Manages the lifecycle of a task graph generated by the LLM.
    """
    def __init__(self, plan_data: Dict[str, Any], logger, context, world_model, planner_mode: Optional[str] = None):
        self.logger = logger
        self.context = context
        self.world_model = world_model
        self.planner_mode = planner_mode
        self.graph = {"nodes": [], "edges": []}
        self.task_states: Dict[str, TaskStatus] = {}
        self.task_outputs: Dict[str, Dict[str, Any]] = {}
        self.dependency_details: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.shared_skill_groups: List[List[str]] = []
        self.produced_params: Dict[str, List[str]] = {}
        self.ready_tasks: List[Dict[str, Any]] = []
        self.task_skill_progress: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.phase_by_task_id: Dict[str, int] = {}
        self.total_phases: int = 0           # Total phase count for this task graph, computed after initialization.

        # Allocation snapshot for recovery after robot faults.
        self._last_allocation_snapshot: Optional[Dict[str, Any]] = None
        self._robot_fault_recovery: bool = False

        # Initialize the atomic task updater.
        self.atomic_task_updater = AtomicTaskUpdater(self.logger, self.context)
        
        self._initialize_graph(plan_data)
        self._compute_phase_info()

    def _initialize_graph(self, plan_data: Dict[str, Any]):
        """Initializes the graph from a pre-validated dictionary."""
        # Extract meta info
        if 'meta' in plan_data and isinstance(plan_data['meta'], dict):
            self.shared_skill_groups = plan_data['meta'].get('shared_skill_groups', [])

        # This logic handles both 'atomic_tasks' and 'task_graph' formats
        if 'task_graph' in plan_data:
            self.graph = plan_data['task_graph']
        elif 'atomic_tasks' in plan_data:
            self.graph['nodes'] = plan_data['atomic_tasks']
            for task in self.graph['nodes']:
                task_id = task.get("task_id")
                for dep_id in task.get("dependencies", []):
                    self.graph['edges'].append({"from": dep_id, "to": task_id, "type": "normal"})
        
        for node in self.graph.get('nodes', []):
            task_id = node.get('task_id')
            if task_id:
                self.task_states[task_id] = TaskStatus.PENDING
                if 'produces' in node and isinstance(node['produces'], list):
                    self.produced_params[task_id] = node['produces']

        # Parse compact edge strings into standard dictionaries.
        raw_edges = self.graph.get('edges', [])
        parsed_edges = [parse_compact_edge(e) for e in raw_edges]
        self.graph['edges'] = parsed_edges

        for edge in parsed_edges:
            from_node = edge.get('from')
            to_node = edge.get('to')
            edge_type = edge.get('type', 'normal')
            
            if from_node and to_node:
                self.dependency_details[to_node].append({
                    "from_task": from_node,
                    "type": edge_type,
                    "condition": edge.get("condition")
                })

        for node in self.graph.get('nodes', []):
            task_id = node.get('task_id')
            if task_id:
                # Use the detailed map to create a simple list of prerequisite task IDs
                prereq_details = self.dependency_details.get(task_id, [])
                node['dependencies'] = [detail['from_task'] for detail in prereq_details]
            else:
                # Ensure the key exists even for malformed nodes
                node['dependencies'] = []

        # Add dependencies for orphan nodes: non-first nodes without incoming edges inherit the previous node's normal incoming edges.
        self._repair_orphan_edges()

    def _repair_orphan_edges(self) -> None:
        """
        Add dependency edges for orphan nodes.

        Rule: for a non-first node without incoming edges, copy the previous node's
        normal incoming edges to it. If the previous node also has no incoming edge,
        resolve the previous node first recursively.
        Added edges are written to self.graph['edges'], self.dependency_details, and node['dependencies'].
        """
        nodes = self.graph.get('nodes', [])
        task_ids = [n.get('task_id') for n in nodes if n.get('task_id')]
        if len(task_ids) <= 1:
            return

        # Collect incoming edge sources for each node.
        has_incoming: Dict[str, List[str]] = {tid: [] for tid in task_ids}
        for edge in self.graph.get('edges', []):
            v = edge.get('to')
            u = edge.get('from')
            if v and u and v in has_incoming:
                has_incoming[v].append(u)

        for idx in range(1, len(task_ids)):
            self._ensure_incoming_edges(idx, task_ids, has_incoming)

    def _ensure_incoming_edges(self, idx: int, task_ids: List[str],
                               has_incoming: Dict[str, List[str]]) -> None:
        """Ensure task_ids[idx] has incoming edges; if not, resolve its predecessor first and inherit its normal incoming edges."""
        tid = task_ids[idx]
        if has_incoming[tid]:
            return  # Already has incoming edges; no processing needed.
        if idx == 0:
            return  # The first node does not need incoming edges.

        # First ensure the previous node has incoming edges.
        self._ensure_incoming_edges(idx - 1, task_ids, has_incoming)

        prev_tid = task_ids[idx - 1]
        # Collect normal incoming edge sources for the previous node.
        prev_normal_sources = []
        for dep in self.dependency_details.get(prev_tid, []):
            if dep.get('type', 'normal') == 'normal':
                prev_normal_sources.append(dep['from_task'])

        # If the previous node has no normal incoming edge, use the previous node itself as a normal dependency.
        if not prev_normal_sources:
            prev_normal_sources = [prev_tid]

        # Add edges for the current orphan node.
        for src in prev_normal_sources:
            new_edge = {"from": src, "to": tid, "type": "normal"}
            self.graph['edges'].append(new_edge)
            self.dependency_details[tid].append({
                "from_task": src,
                "type": "normal",
                "condition": None
            })
            has_incoming[tid].append(src)

        # Update the node dependencies list.
        node = self.get_task_by_id(tid)
        if node:
            node['dependencies'] = [d['from_task'] for d in self.dependency_details.get(tid, [])]

    def _compute_phase_info(self) -> None:
        """
        Compute the plan phase count from dependencies across the whole task graph.

        Rules:
        - All tasks start at phase 1.
        - For each edge u -> v:
            * If type == 'conditional', v must be at least one phase after u: phase[v] >= phase[u] + 1.
            * Otherwise, such as normal, v must be in the same phase as u or later: phase[v] >= phase[u].
        - Orphan node incoming edges are added in _repair_orphan_edges; no extra processing is needed here.
        - Run one topological DP pass on the DAG. The final total_phases = max(phase.values()).
        """
        task_ids = [node.get("task_id") for node in self.graph.get("nodes", []) if node.get("task_id")]
        if not task_ids:
            self.total_phases = 0
            self.phase_by_task_id = {}
            return

        from collections import defaultdict, deque

        succ: dict[str, list[tuple[str, str]]] = defaultdict(list)
        indeg: dict[str, int] = {tid: 0 for tid in task_ids}

        for edge in self.graph.get("edges", []):
            u = edge.get("from")
            v = edge.get("to")
            etype = edge.get("type", "normal")
            if not u or not v:
                continue
            if u not in indeg or v not in indeg:
                continue
            succ[u].append((v, etype))
            indeg[v] += 1

        # All tasks are at least in phase 1.
        level: dict[str, int] = {tid: 1 for tid in task_ids}

        q = deque([tid for tid in task_ids if indeg[tid] == 0])
        if not q:
            q = deque(task_ids)

        visited = set()

        while q:
            u = q.popleft()
            visited.add(u)
            base_level = level[u]

            for v, etype in succ.get(u, []):
                delta = 1 if etype == "conditional" else 0
                if level[v] < base_level + delta:
                    level[v] = base_level + delta

                indeg[v] -= 1
                if indeg[v] == 0 and v not in visited:
                    q.append(v)

        # Record phase information.
        self.phase_by_task_id = level
        self.total_phases = max(1, max(level.values()) if level else 1)

    def get_ready_tasks(self) -> List[Dict[str, Any]]:
        """
        Identifies all tasks that are ready to be dispatched for allocation.

        - phase mode: dispatch all nodes in the graph at once.
        - full mode: dispatch in batches by phase.
          1. Find the smallest current phase that still has PENDING tasks.
          2. Check whether all PENDING tasks in that phase have no tbd parameters.
             - If none have tbd, all PENDING tasks in that phase become ready_tasks.
             - If any have tbd, upstream output has not been filled yet, so return an empty list and wait.
        """
        # phase: all nodes in the graph are ready, deep-copied to avoid modifying the original graph.
        if getattr(self, "planner_mode", None) == "phase":
            self.ready_tasks = [deepcopy(n) for n in self.graph.get('nodes', []) if n.get('task_id')]
            for node in self.ready_tasks:
                tid = node['task_id']
                if self.task_states.get(tid) == TaskStatus.PENDING:
                    self.task_states[tid] = TaskStatus.READY
            self._prune_batch_dependencies()
            return self.ready_tasks

        # full: dispatch in batches by phase.
        if not self.phase_by_task_id:
            self.ready_tasks = []
            return self.ready_tasks

        # 1. Collect all tasks that are still PENDING and their phases.
        pending_phases: Dict[int, List[str]] = defaultdict(list)
        for tid, status in self.task_states.items():
            if status == TaskStatus.PENDING:
                phase = self.phase_by_task_id.get(tid, 1)
                pending_phases[phase].append(tid)

        if not pending_phases:
            self.ready_tasks = []
            return self.ready_tasks

        # 2. Take the smallest phase.
        min_phase = min(pending_phases.keys())
        candidate_ids = pending_phases[min_phase]

        # 3. Check whether any task in this phase has tbd parameters.
        has_tbd = False
        for tid in candidate_ids:
            task_node = self.get_task_by_id(tid)
            if task_node and self._has_tbd_params(task_node):
                has_tbd = True
                break

        if has_tbd:
            # This phase has unfilled tbd parameters, so wait for upstream outputs.
            self.ready_tasks = []
            return self.ready_tasks

        # 4. All PENDING tasks in this phase can be dispatched.
        self.ready_tasks = [deepcopy(self.get_task_by_id(tid))
                            for tid in candidate_ids if self.get_task_by_id(tid)]

        for task in self.ready_tasks:
            self.task_states[task['task_id']] = TaskStatus.READY

        self._prune_batch_dependencies()
        return self.ready_tasks
    
    def get_ready_tasks_dependencies(self) -> List[Tuple[str, str]]:
        """
        Calculates the dependency relationships *only* between tasks
        that are in the current `ready_tasks` batch.
        """
        if not self.ready_tasks:
            return []

        ready_task_ids = {task['task_id'] for task in self.ready_tasks}
        dependencies_within_batch: List[Tuple[str, str]] = []

        # Iterate through the ready tasks to find dependencies pointing to them
        for task in self.ready_tasks:
            to_task = task['task_id']
            for dep_detail in self.dependency_details.get(to_task, []):
                from_task = dep_detail['from_task']
                if from_task in ready_task_ids:
                    dependencies_within_batch.append((from_task, to_task))

        return dependencies_within_batch
    
    def get_dispatchable_skill_groups(self, ready_tasks: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Filters shared_skill_groups for the current batch of ready tasks.

        - If a group is fully contained in ready_task_ids, dispatch it as-is.
        - If only partially overlapping, split: the ready portion (at least 2 refs) is dispatched,
          the remainder (at least 2 refs) is kept in self.shared_skill_groups for future batches.
        - Single-ref remainders are discarded (no grouping needed).
        """
        if not self.shared_skill_groups:
            return []

        ready_task_ids = {task['task_id'] for task in ready_tasks}
        dispatchable_groups = []
        retained_groups = []

        for group in self.shared_skill_groups:
            group_task_ids = {ref.split('.')[0] for ref in group}

            if group_task_ids.issubset(ready_task_ids):
                # Fully ready - dispatch entire group.
                dispatchable_groups.append(group)
            else:
                # Partially ready - split.
                ready_refs = [ref for ref in group if ref.split('.')[0] in ready_task_ids]
                remaining_refs = [ref for ref in group if ref.split('.')[0] not in ready_task_ids]

                if len(ready_refs) >= 2:
                    dispatchable_groups.append(ready_refs)
                if len(remaining_refs) >= 2:
                    retained_groups.append(remaining_refs)

        # Update shared_skill_groups: keep only the retained (future) portions
        self.shared_skill_groups = retained_groups

        return dispatchable_groups
    
    def _prune_batch_dependencies(self) -> None:
        """Keep only dependencies inside the current batch; operates on self.ready_tasks copies and does not affect the original graph."""
        if not self.ready_tasks:
            return
        ready_ids = {t.get('task_id') for t in self.ready_tasks if t.get('task_id')}
        for task in self.ready_tasks:
            deps = task.get('dependencies') or []
            task['dependencies'] = [d for d in deps if d in ready_ids]

    # =========================================================================
    # Allocation Snapshot and Robot Fault Recovery
    # =========================================================================

    def save_allocation_snapshot(
        self,
        ready_tasks: List[Dict[str, Any]],
        dispatchable_groups: List[List[str]],
    ) -> None:
        """Save this allocation input snapshot for recovery after robot faults.

        Called by PlanningLayer after allocation succeeds.
        robot_view is read from context._generated_text["alloc_results"]["robot_view"].
        """
        alloc_results = (
            getattr(self.context, "_generated_text", {}) or {}
        ).get("alloc_results") or {}
        robot_view = alloc_results.get("robot_view") or {}

        self._last_allocation_snapshot = {
            "ready_tasks": deepcopy(ready_tasks),
            "dispatchable_groups": deepcopy(dispatchable_groups),
            "robot_view": deepcopy(robot_view),
        }

    def _handle_robot_fault_recovery(self, details: Dict[str, Any]) -> None:
        """Recovery logic for robot fault, low battery, or communication interruption.

        1. Extract the failed robot label from details.
        2. Find this robot's {task_id: [skill_list]} from snapshot robot_view.
        3. Roll back skill progress for skills handled by the failed robot.
        4. Revert affected task states.
        5. Set the recovery mark.
        """
        snapshot = self._last_allocation_snapshot
        if not snapshot:
            # No snapshot; fall back to the original coarse-grained logic.
            self._handle_robot_fault_fallback()
            return

        # 1. Extract failed robot label.
        robot_info = details.get("robot") or {}
        failed_label = (
            robot_info.get("label")
            or details.get("robot_label")
            or details.get("payload", {}).get("robot_label")
        )
        if not failed_label:
            dlog("[TaskGraphManager] Cannot identify failed robot label, falling back.",
                 logger=self.logger, level="warning")
            self._handle_robot_fault_fallback()
            return

        robot_view = snapshot.get("robot_view") or {}

        # 2. Find the failed robot's {task_id: [skill_name_list]}.
        failed_robot_skills: Dict[str, List[str]] = robot_view.get(failed_label) or {}

        # 3. Roll back skill progress for the failed robot only; keep other robots' progress.
        for task_id, skill_names in failed_robot_skills.items():
            progress = self.task_skill_progress.get(task_id)
            if not progress:
                continue
            for skill_name in skill_names:
                base = self._parse_base_skill_name(skill_name)
                if base:
                    progress.pop(base, None)

        # 4. Revert task states.
        for task_id, prev_state in list(self.task_states.items()):
            if prev_state in (TaskStatus.DISPATCHED, TaskStatus.READY, TaskStatus.FAILED):
                # Revert non-completed tasks directly.
                self.task_states[task_id] = TaskStatus.PENDING
            elif prev_state == TaskStatus.COMPLETED and task_id in failed_robot_skills:
                # Completed tasks: check whether they still satisfy all completion conditions after rollback.
                task_node = self.get_task_by_id(task_id)
                if not task_node:
                    continue
                required_skills_list = task_node.get("required_skills", []) or []
                required_skill_set = set()
                for s in required_skills_list:
                    if isinstance(s, str):
                        parsed = parse_compact_skill(s)
                        name = parsed.get("skill_name", "")
                    else:
                        name = s.get("skill_name", "")
                    base = self._parse_base_skill_name(name)
                    if base:
                        required_skill_set.add(base)
                progress = self.task_skill_progress.get(task_id, {})
                completed_set = {name for name, ok in progress.items() if ok}
                if not required_skill_set.issubset(completed_set):
                    # No longer satisfies all completion conditions, so revert to PENDING.
                    self.task_states[task_id] = TaskStatus.PENDING
                    dlog(f"Task {task_id} reverted to PENDING due to robot fault recovery.",
                         logger=self.logger)

        # 5. Set recovery mark.
        self._robot_fault_recovery = True
        dlog(f"[TaskGraphManager] Robot fault recovery prepared for '{failed_label}'. "
             f"Affected tasks: {list(failed_robot_skills.keys())}",
             logger=self.logger)

    def _handle_robot_fault_fallback(self) -> None:
        """Coarse-grained revert when no snapshot exists; original logic."""
        for task_id, prev_state in list(self.task_states.items()):
            if prev_state in (TaskStatus.DISPATCHED, TaskStatus.READY, TaskStatus.FAILED):
                self.task_states[task_id] = TaskStatus.PENDING
                self.task_skill_progress.pop(task_id, None)

    def get_ready_tasks_for_recovery(self) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """Robot fault recovery only: recover allocation input from snapshot and trim completed skills.

        Returns:
            (ready_tasks, dispatchable_groups) - trimmed allocation input.
        """
        snapshot = self._last_allocation_snapshot
        self._robot_fault_recovery = False

        if not snapshot:
            return self.get_ready_tasks(), []

        ready_tasks = deepcopy(snapshot["ready_tasks"])
        dispatchable_groups = deepcopy(snapshot["dispatchable_groups"])

        # Record removed skill references for synchronized dispatchable_groups cleanup.
        removed_skill_refs: Set[str] = set()
        filtered_tasks: List[Dict[str, Any]] = []

        for task in ready_tasks:
            task_id = task.get("task_id")
            if not task_id:
                continue

            # If the task is still COMPLETED and was not reverted, skip it.
            if self.task_states.get(task_id) == TaskStatus.COMPLETED:
                # All skills for this task are complete, so mark all references as removed.
                for idx in range(len(task.get("required_skills", []))):
                    removed_skill_refs.add(f"{task_id}.{idx}")
                continue

            # Trim completed skills.
            progress = self.task_skill_progress.get(task_id, {})
            original_skills = task.get("required_skills", []) or []
            kept_skills: List[Any] = []
            for idx, sk in enumerate(original_skills):
                if isinstance(sk, str):
                    parsed = parse_compact_skill(sk)
                    name = parsed.get("skill_name", "")
                else:
                    name = sk.get("skill_name", "")
                base = self._parse_base_skill_name(name)
                if base and progress.get(base) is True:
                    # This skill was completed by a non-failed robot, so remove it.
                    removed_skill_refs.add(f"{task_id}.{idx}")
                else:
                    kept_skills.append(sk)

            if not kept_skills:
                # All skills are complete; this should not happen because COMPLETED was checked above.
                for idx in range(len(original_skills)):
                    removed_skill_refs.add(f"{task_id}.{idx}")
                continue

            task["required_skills"] = kept_skills
            # Set task state to READY.
            self.task_states[task_id] = TaskStatus.READY
            filtered_tasks.append(task)

        # Remove deleted skill references from dispatchable_groups.
        cleaned_groups: List[List[str]] = []
        for group in dispatchable_groups:
            cleaned = [ref for ref in group if ref not in removed_skill_refs]
            if len(cleaned) >= 2:
                cleaned_groups.append(cleaned)

        # Update ready_tasks reference.
        self.ready_tasks = filtered_tasks
        self._prune_batch_dependencies()

        dlog(f"[TaskGraphManager] Recovery: {len(filtered_tasks)} tasks, "
             f"{len(cleaned_groups)} skill groups, "
             f"{len(removed_skill_refs)} skill refs removed.",
             logger=self.logger)

        return filtered_tasks, cleaned_groups

    def update_from_feedback(self, feedback_processor):
        """
        Update the task graph using FeedbackProcessor.last_event.
        This unified entry point can process newcase events and normal execution results.
        """
        last_event = feedback_processor.last_event
        if not last_event:
            return

        details = last_event.get('details', {})
        outcomes = details.get('outcomes')
        event_kind = last_event.get('event_kind')

        if not event_kind and outcomes is not None:
            # Process normal execution results.
            self._update_task_statuses(outcomes)
            self._fill_tbd_from_runtime_params(outcomes)
        else:
            # Process newcase events.
            event_type = last_event.get('type')
            self.atomic_task_updater.update_tasks_for_event(
                self.graph['nodes'], event_type, details
            )
            # Robot-related newcases: fine-grained recovery at skill level.
            if event_type in ("ROBOT_FAULT", "ROBOT_BATTERY_LOW", "ROBOT_COMM_JAMMED"):
                self._handle_robot_fault_recovery(details)

    def _update_task_statuses(self, outcomes: List[Dict[str, Any]]) -> Set[str]:
        """
        Update task completion status from incremental outcomes.
        Supports multiple calls and processes only newly added skill results each time.
        
        Returns the task_id set marked as COMPLETED in this call.
        """
        if not outcomes:
            return set()

        # Record task_ids touched by this batch of outcomes.
        updated_task_ids: Set[str] = set()

        # 1. Incrementally update skill completion for each task.
        for outcome in outcomes:
            meta = outcome.get("meta", {}) or {}
            task_id = meta.get("task_id")
            raw_skill_name = (meta.get("skill") or "").strip()

            if not task_id or not raw_skill_name:
                continue
            if task_id not in self.task_states:
                continue

            base_skill_name = self._parse_base_skill_name(raw_skill_name)
            if not base_skill_name:
                continue

            # Prefer success state from meta, then data, defaulting to True.
            success = meta.get("success")
            if success is None:
                success = (outcome.get("data", {}) or {}).get("success", True)

            # Update cache: one success counts as success; failures do not overwrite previous success.
            prev = self.task_skill_progress[task_id].get(base_skill_name)
            if prev is True:
                # Already recorded as success; no update needed.
                pass
            else:
                self.task_skill_progress[task_id][base_skill_name] = bool(success)

            updated_task_ids.add(task_id)

        # 2. For updated tasks, check whether they can be marked COMPLETED; all base skills must succeed.
        just_completed: Set[str] = set()
        for task_id in updated_task_ids:
            # Check only incomplete tasks; completed tasks do not need repeated checks.
            if self.task_states.get(task_id) == TaskStatus.COMPLETED:
                continue

            task_node = self.get_task_by_id(task_id)
            if not task_node:
                continue

            # All base skill names required by the task definition.
            required_skills_list = task_node.get("required_skills", []) or []
            required_skill_set = set()
            for s in required_skills_list:
                if isinstance(s, str):
                    parsed = parse_compact_skill(s)
                    name = parsed.get("skill_name", "")
                else:
                    name = s.get("skill_name", "")
                base = self._parse_base_skill_name(name)
                if base:
                    required_skill_set.add(base)

            if not required_skill_set:
                continue

            # Accumulated completed skill set across multiple outcome calls.
            progress_map = self.task_skill_progress.get(task_id, {}) or {}
            completed_skill_set = {name for name, ok in progress_map.items() if ok}

            # Mark the task complete only when all required skills succeeded.
            if required_skill_set.issubset(completed_skill_set):
                if self.task_states.get(task_id) != TaskStatus.COMPLETED:
                    self.task_states[task_id] = TaskStatus.COMPLETED
                    dlog(f"Task {task_id} marked as COMPLETED.", logger=self.logger)
                    just_completed.add(task_id)

        # 3. For just-completed tasks, try to fill TBD parameters using only related outcomes.
        if just_completed:
            relevant_outcomes = [
                o for o in outcomes
                if (o.get("meta", {}) or {}).get("task_id") in just_completed
            ]
            if relevant_outcomes:
                self._fill_tbd_from_runtime_params(relevant_outcomes)

        return just_completed

    def _parse_base_skill_name(self, skill_string: str) -> str:
        """Extract the base skill name from a parameterized skill string, such as 'search<p-2>_for<car>' -> 'search'."""
        if not skill_string:
            return ""
        # The base skill name is the part before the first '<'.
        return skill_string.split('<')[0].strip()

    def _fill_tbd_from_runtime_params(self, outcomes: List[Dict]):
        """
        Internal helper that fills TBD placeholders in downstream tasks from completed task produces and runtime parameters.
        """
        if not self.world_model or not outcomes:
            return
            
        runtime_params = self.context._generated_text.get("runtime_params", {})
        pipeline_hints = runtime_params.get("pipeline_hints", {})
        if not pipeline_hints:
            return
        
        # 1. Determine replacement values.
        target_value = None
        target_id = pipeline_hints.get("selected_target_id")
        if target_id:
            entity_node = self.world_model.get_node_by_id(target_id)
            if entity_node:
                target_value = entity_node.get("properties", {}).get("label")

        found_ids = (runtime_params.get('last') or {}).get('found_ids', [])
        # count_value = len(found_ids) if found_ids else 1 # Optional implementation.
        count_value = 1 # Currently set to 1.

        # 2. Build the parameter dictionary to fill from completed task produces lists.
        params_to_fill = {}
        completed_task_ids = {task for task, status in self.task_states.items() if status == TaskStatus.COMPLETED}
        
        # Collect tbd placeholder names from all PENDING tasks for exact fallback matching.
        pending_tbd_names = set()
        for node in self.graph.get('nodes', []):
            tid = node.get('task_id')
            if tid and self.task_states.get(tid) == TaskStatus.PENDING:
                node_str = json.dumps(node)
                for match in re.findall(r'tbd:(\w+)', node_str):
                    pending_tbd_names.add(match)

        for task_id in completed_task_ids:
            produced_vars = self.produced_params.get(task_id, [])
            for var_name in produced_vars:
                # Use heuristic rules to determine parameter type.
                if any(kw in var_name.lower() for kw in ["location", "spot", "position", "point", "area", "target", "object"]):
                    if target_value:
                        params_to_fill[var_name] = target_value
                elif any(kw in var_name.lower() for kw in ["count", "number", "quantity"]):
                    params_to_fill[var_name] = count_value
                elif var_name in pending_tbd_names and target_value:
                    # Fallback: if the produces variable exactly matches a tbd placeholder, fill it with target_value.
                    params_to_fill[var_name] = target_value
                    dlog(f"Fallback TBD fill: '{var_name}' matched pending placeholder, using target_value.", logger=self.logger)
        
        if not params_to_fill:
            return

        # 3. Apply these updates to all PENDING tasks in the graph.
        self._apply_parameter_updates(params_to_fill)

    def _apply_parameter_updates(self, params_to_update: Dict[str, Any]):
        """
        Scan and replace TBD placeholders in PENDING tasks using the given parameter dictionary.
        """
        for i, node in enumerate(self.graph['nodes']):
            task_id = node.get('task_id')
            # Update only tasks that have not been dispatched.
            if not task_id or self.task_states.get(task_id) != TaskStatus.PENDING:
                continue

            node_str = json.dumps(node)
            if 'tbd:' not in node_str:
                continue
            
            original_node_str = node_str
            for param_name, param_value in params_to_update.items():
                placeholder = f"tbd:{param_name}"
                if placeholder not in node_str:
                    continue

                # Use json.dumps to handle string and number replacements correctly.
                # e.g., for strings: "tbd:..." -> "\"value\""
                # e.g., for numbers: "tbd:..." -> "1"
                value_as_json_fragment = json.dumps(param_value)

                # Replace forms like "key": "tbd:param_name".
                node_str = node_str.replace(f'"{placeholder}"', value_as_json_fragment)
                
                # Replace forms like "skill<tbd:param_name>".
                node_str = node_str.replace(f'<{placeholder}>', f'<{str(param_value)}>')

            # If replacement happened, update the node in the graph.
            if original_node_str != node_str:
                try:
                    updated_node = json.loads(node_str)
                    self.graph['nodes'][i] = updated_node
                    dlog(f"Updated task node {task_id} with runtime parameters.", logger=self.logger)
                except json.JSONDecodeError:
                    dlog(f"Failed to decode updated task node for task {task_id}. Skipping update.", logger=self.logger, level="error")
    
    def mark_as_dispatched(self, task_ids: List[str]):
        """Marks a list of tasks as dispatched for execution."""
        for task_id in task_ids:
            if self.task_states.get(task_id) == TaskStatus.READY:
                self.task_states[task_id] = TaskStatus.DISPATCHED

    def is_complete(self) -> bool:
        """Checks if all tasks in the graph have been completed."""
        return all(status == TaskStatus.COMPLETED for status in self.task_states.values())
    
    def has_pending_tasks(self) -> bool:
        # Returns True as soon as the first PENDING task is found.
        return any(status == TaskStatus.PENDING for status in self.task_states.values())

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        for node in self.graph.get('nodes', []):
            if node.get('task_id') == task_id:
                return node
        return None
    
    def get_completion_stats(self) -> Dict[str, int]:
        """Get atomic task completion statistics."""
        total_tasks = len(self.graph.get('nodes', []))
        if total_tasks == 0:
            return {"completed": 0, "total": 0}

        completed_count = sum(1 for status in self.task_states.values() if status == TaskStatus.COMPLETED)
        return {"completed": completed_count, "total": total_tasks}

    def _has_tbd_params(self, task: Dict[str, Any]) -> bool:
        """Recursively checks for 'tbd:' strings in a task dictionary."""
        return "tbd:" in json.dumps(task)
