import json
import re
import traceback
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict

from modules.task_solver.llm_framework.core.parser import parse_text
from modules.task_solver.sgi_planner.utils.compact_parsers import parse_compact_skill, parse_compact_edge


# =====================
# Common Parsing and Light Auto-Fix
# =====================

def _loads_try(s: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Try to json.loads a string, returning (success, error message, data)."""
    try:
        return True, "", json.loads(s)
    except json.JSONDecodeError as e:
        return False, str(e), {}


def parse_json_with_auto_fix(raw: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Try to parse LLM JSON output while allowing common errors:
    - Wrapped in ```json ... ```
    - Single quotes instead of double quotes
    - Extra trailing commas
    Returns: (success, error message, parsed data)
    """
    text = raw.strip()
    # text = parse_text(text, "json", all_matches=False)

    # Remove markdown fence.
    fence_match = re.match(r"^```(?:json)?\s*(.*)```$", text, re.S | re.I)
    if fence_match:
        text = fence_match.group(1).strip()

    ok, err, data = _loads_try(text)
    if ok:
        return ok, "", data

    # Try automatic fixes.
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)  # Remove trailing commas.
    if fixed.count('"') < fixed.count("'"):      # Weak fix: single quotes -> double quotes.
        fixed = re.sub(r"'", r'"', fixed)

    ok, err2, data = _loads_try(fixed)
    if ok:
        return ok, "", data

    return False, f"Invalid JSON format: {err2}", {}

def _clean_tbd(v):
    """Remove the tbd: string prefix."""
    return v[4:] if isinstance(v, str) and v.startswith("tbd:") else v


def _clean_produces(task: Dict[str, Any]) -> None:
    prods = task.get("produces")
    if isinstance(prods, str):
        task["produces"] = _clean_tbd(prods)
    elif isinstance(prods, list):
        task["produces"] = [_clean_tbd(x) for x in prods]


def _clean_edge(edge: Dict[str, Any]) -> None:
    cond = edge.get("condition")
    if isinstance(cond, str):
        edge["condition"] = cond.replace("tbd:", "")


def adapt_plan_to_universal_format(plan_data: Dict[str, Any]) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Adapt different plan formats to a unified internal representation.
    Returns: (ok, err, tasks_list, meta_data)
    """
    meta_data = plan_data.get("meta")
    if meta_data is not None and not isinstance(meta_data, dict):
        return False, "'meta' key must correspond to a dictionary", [], None

    # -------- Format 1: atomic_tasks --------
    if "atomic_tasks" in plan_data and isinstance(plan_data["atomic_tasks"], list):
        tasks = plan_data["atomic_tasks"]
        for t in tasks:
            if isinstance(t, dict):
                _clean_produces(t)
        return True, "", tasks, meta_data

    # -------- Format 2: task_graph --------
    if "task_graph" in plan_data and isinstance(plan_data["task_graph"], dict):
        graph = plan_data["task_graph"]
        nodes, edges = graph.get("nodes"), graph.get("edges")

        if not isinstance(nodes, list):
            return False, "'task_graph' missing 'nodes' list", [], meta_data
        if not isinstance(edges, list):
            return False, "'task_graph' missing 'edges' list", [], meta_data

        # Clean nodes.produces.
        for node in nodes:
            if isinstance(node, dict):
                _clean_produces(node)

        # Parse compact edge strings to standard dictionaries, then clean condition.
        parsed_edges = []
        for edge in edges:
            parsed = parse_compact_edge(edge)
            _clean_edge(parsed)
            parsed_edges.append(parsed)
        graph["edges"] = parsed_edges

        # Build dependencies.
        deps_map = defaultdict(list)
        for edge in parsed_edges:
            if isinstance(edge, dict) and "from" in edge and "to" in edge:
                deps_map[edge["to"]].append(edge["from"])

        tasks_with_deps = []
        for node in nodes:
            if isinstance(node, dict):
                tid = node.get("task_id")
                node["dependencies"] = deps_map.get(tid, [])
                tasks_with_deps.append(node)

        return True, "", tasks_with_deps, meta_data

    return False, "Unrecognized plan format, top level should contain 'atomic_tasks' or 'task_graph'", [], meta_data

# =====================
# Schema Fixes and Normalization
# =====================

def fix_schema_defaults(tasks: List[Dict[str, Any]]) -> None:
    """
    Fix the unified task list in place.
    """
    for t in tasks:
        if not isinstance(t, dict):
            continue

        if "task_id" in t and not isinstance(t["task_id"], str):
            t["task_id"] = str(t["task_id"])

        t.setdefault("dependencies", [])
        if isinstance(t["dependencies"], (int, float)):
            t["dependencies"] = [str(t["dependencies"])]
        if isinstance(t["dependencies"], str):
            t["dependencies"] = [t["dependencies"]]

        if "required_skills" not in t or t["required_skills"] is None:
            t["required_skills"] = []
        if isinstance(t["required_skills"], dict):
            t["required_skills"] = [t["required_skills"]]
        if isinstance(t["required_skills"], str):
            t["required_skills"] = [parse_compact_skill(t["required_skills"])]

        # Parse compact skill strings in the list into dictionaries.
        t["required_skills"] = [
            parse_compact_skill(sk) if isinstance(sk, str) else sk
            for sk in t["required_skills"]
        ]


def fix_dependencies_type(tasks: List[Dict[str, Any]]) -> None:
    """
    Fix dependency item types to strings.
    """
    for t in tasks:
        deps = t.get("dependencies", [])
        if isinstance(deps, list):
            t["dependencies"] = [str(d) if isinstance(d, (int, float)) else d for d in deps]
        else:
            # Already handled in fix_schema_defaults; keep this as a guard.
            t["dependencies"] = []


# =====================
# Skill-Related Fixes and Validation
# =====================

def _iterate_all_skills(skills_data: Dict[str, Any]):
    """
    Iterate over all skill definitions for one robot entry in the skill library.
    Compatible with:
    {
      "basic_skills": {...},
      "integrated_skills": {...}
    }
    Or directly { "navigate": {...}, ... }
    """
    if "basic_skills" in skills_data or "integrated_skills" in skills_data:
        if "basic_skills" in skills_data and isinstance(skills_data["basic_skills"], dict):
            for v in skills_data["basic_skills"].values():
                yield v
        if "integrated_skills" in skills_data and isinstance(skills_data["integrated_skills"], dict):
            for v in skills_data["integrated_skills"].values():
                yield v
    else:
        for v in skills_data.values():
            yield v


def extract_skills_from_library(robot_skill_library: Dict[str, Any]) -> Set[str]:
    """
    Extract all template names from the skill library, such as
    'navigate<location>' or 'place<object>_on<surface>'.
    """
    skills: Set[str] = set()
    for _, skills_data in robot_skill_library.items():
        for skill_info in _iterate_all_skills(skills_data):
            name = skill_info.get("name")
            if isinstance(name, str):
                skills.add(name)
    return skills


def remove_robot_prefix(skill_name: str) -> str:
    """
    'Humanoid.navigate<Hotel-1>' -> 'navigate<Hotel-1>'
    'UGV.place<...>' -> 'place<...>'
    """
    m = re.match(r"^[A-Za-z][A-Za-z0-9_-]*\.(.+)$", skill_name)
    return m.group(1) if m else skill_name


def _place_repl(m) -> str:
    obj = m.group(1).strip("<>")
    dst = m.group(2).strip("<>")
    return f"place<{obj}>_on<{dst}>"


def _normalize_place_phrasing(name: str) -> str:
    """
    place<obj>_on<dst> / place<obj>_onto<dst> / place obj on dst
    => place<obj>_on<dst>
    """
    pat = r"place(<[^>]+>|[^<>\s_]+)[\s_]*o?n?to?[\s_]*(<[^>]+>|[^<>\s_]+)"
    return re.sub(pat, _place_repl, name, flags=re.I)


def extract_base_skill_name(skill_name: str) -> str:
    """
    'search<area>_for<target>' -> 'search'
    'place<object>_to<surface>' -> 'place'
    'navigate<location>' -> 'navigate'
    """
    cleaned = re.sub(r"<[^>]*>", "", skill_name)
    return cleaned.split("_")[0].split()[0]


def _guide_repl(m) -> str:
    who = m.group(1).strip("<>")
    loc = m.group(2).strip("<>")
    return f"guide<{who}>_to<{loc}>"


def _search_repl(m) -> str:
    area = m.group(1).strip("<>")
    tgt = m.group(2).strip("<>")
    return f"search<{area}>_for<{tgt}>"


def fix_skill_name_format(skill_name: str) -> str:
    """
    Syntax fixes for a single skill name:
    - Remove robot prefixes such as 'UGV.' or 'Humanoid.'
    - place ... on/onto -> _to<>
    - search area for target -> search<area>_for<target>
    - guide person to loc -> guide<person>_to<loc>
    - navigate Hotel-1 -> navigate<Hotel-1>
    - Ensure coordinates [[...]] are wrapped in <...>
    """
    name = remove_robot_prefix(skill_name).strip()

    # Wrap coordinates [[...]] -> <[[...]]>.
    name = re.sub(
        r'(?<![<>])(\[\[[\d\s.,\[\]-]+\]\])(?![<>])',
        r'<\1>',
        name
    )

    # Apply skill-specific fix rules based on the verb.
    base_verb = extract_base_skill_name(name)

    if base_verb == "place":
        name = _normalize_place_phrasing(name)
    
    elif base_verb == "guide":
        pat_guide = r"guide(<[^>]+>|[^<>\s_]+)[\s_]*(?:o?n?to?)[\s_]*(<[^>]+>|[^<>\s_]+)"
        name = re.sub(pat_guide, _guide_repl, name, flags=re.I)

    elif base_verb == "search":
        pat_search = r"search(<\[[^\]]+\]>|[^<>\s_]+)[\s_]*for[\s_]*(<[^>]+>|[^<>\s_]+)"
        name = re.sub(pat_search, _search_repl, name, flags=re.I)

    # Handle single-parameter skills.
    single_param_skills = ["navigate", "take_photo", "follow", "broadcast", "handle_hazard"]
    for sk in single_param_skills:
        pat = rf"\b{sk}\b(?!<)[\s_]+([^<>\s_][^<>\s]*)"
        name = re.sub(pat, rf"{sk}<\1>", name)

    # Clean possible duplicate angle brackets.
    name = re.sub(r"<{2,}", "<", name)
    name = re.sub(r">{2,}", ">", name)

    return name


def fix_skill_name_with_context(skill_name: str, task_node: Dict[str, Any]) -> str:
    """
    If placeholders such as <location> or <object> remain, fill them from the task description:
    - Use coordinates like [[...],[...]] for location placeholders.
    - Use IDs like 'Thing-1' for entity placeholders.
    """
    name = fix_skill_name_format(skill_name)
    placeholders = re.findall(r"<([^>]+)>", name)
    desc = task_node.get("description", "") if isinstance(task_node, dict) else ""

    for ph in placeholders:
        ph_low = ph.lower()
        replacement = None

        generic_loc = ["area", "region", "location", "zone", "position"]
        generic_obj = ["target", "object", "item", "cargo", "hazard", "person", "surface"]

        # Location placeholders: try coordinates [[...]].
        if ph_low in generic_loc:
            m_coord = re.search(r"(\[\[[\d\s.,\[\]-]+\]\])", desc)
            if m_coord:
                replacement = m_coord.group(1)

        # Entity placeholders: try 'Word-123'.
        if not replacement and ph_low in (generic_obj + generic_loc):
            m_ent = re.search(r"([A-Za-z]+-\d+)", desc)
            if m_ent:
                replacement = m_ent.group(1)

        if replacement:
            name = name.replace(f"<{ph}>", f"<{replacement}>")

    return name


def validate_parameterized_skill(skill_name: str, available_skills: Set[str]) -> Tuple[bool, str]:
    """
    Check whether the skill name is in the skill library, allowing concrete parameters.
    Returns (is valid, final skill_name).
    """
    candidate = fix_skill_name_format(skill_name)

    # 1. Direct full match.
    if candidate in available_skills:
        return True, candidate

    # 2. Same base skill name is enough to consider it instantiable.
    base_candidate = extract_base_skill_name(candidate)
    lib_bases = {extract_base_skill_name(s) for s in available_skills}
    if base_candidate in lib_bases:
        return True, candidate

    # 3. Same base skill as a library template with <>.
    for tmpl in available_skills:
        if "<" in tmpl and ">" in tmpl:
            if extract_base_skill_name(tmpl) == base_candidate:
                return True, candidate

    return False, skill_name

def find_text_position(text: str, search_str: str, start_from: int = 0) -> Tuple[int, int]:
    """
    Find the position range of a string in text.
    
    Args:
        text: Text to search.
        search_str: String to find.
        start_from: Position to start searching from.
    
    Returns:
        (start_pos, end_pos), or (-1, -1) if not found.
    """
    pos = text.find(search_str, start_from)
    if pos == -1:
        return -1, -1
    return pos, pos + len(search_str) - 1

def fix_and_validate_skills(tasks: List[Dict[str, Any]], robot_skill_library: Dict[str, Any], plan_str: str = None, enable_fix: bool = True) -> Tuple[bool, List[str]]:
    """
    Iterate through all required_skills and try to:
    - Convert string skills to { "skill_name": ... }.
    - Normalize skill_name.
    - Fill parameters from the description.
    - Validate against the skill library.
    - Update skill_name in place on success.
    Returns: (whether all are valid, error list).
    """
    errors: List[str] = []
    available_skills = extract_skills_from_library(robot_skill_library)

    for ti, task in enumerate(tasks):
        skills_list = task.get("required_skills", [])
        if not isinstance(skills_list, list):
            continue

        for si, skill in enumerate(skills_list):
            skill = parse_compact_skill(skill)
            skills_list[si] = skill

            if not isinstance(skill, dict):
                errors.append(f"Task {ti}, skill {si}: must be a dict")
                continue

            raw_name = skill.get("skill_name")
            if not isinstance(raw_name, str):
                errors.append(f"Task {ti}, skill {si}: missing 'skill_name'")
                continue

            # Check robot prefixes such as Quadruped.xxx or UAV.xxx.
            robot_prefix_pattern = r"^[A-Za-z][A-Za-z0-9_-]*\."
            if re.match(robot_prefix_pattern, raw_name):
                if not enable_fix:
                    # When automatic fixes are not allowed, treat this as a schema error.
                    position = -1
                    if plan_str and raw_name:
                        search_patterns = [
                            f'"{raw_name}"',
                            f"'{raw_name}'",
                            raw_name
                        ]
                        for pattern in search_patterns:
                            start_pos, end_pos = find_text_position(plan_str, pattern)
                            if start_pos != -1:
                                if pattern.startswith('"') or pattern.startswith("'"):
                                    position = [start_pos + 1, end_pos - 1]
                                else:
                                    position = [start_pos, end_pos]
                                break

                    errors.append({
                        "message": f"Task {ti}, skill {si}: skill_name should not contain robot prefix form '{raw_name}'",
                        "position": position,
                        "error_type": "invalid_schema_robot_prefix"
                    })
                    continue

            contextual_name = fix_skill_name_with_context(raw_name, task)
            ok, final_name = validate_parameterized_skill(contextual_name, available_skills)

            if ok:
                # Write back only when fixes are enabled.
                if enable_fix:
                    skill["skill_name"] = fix_skill_name_format(final_name)
            else:
                # Try to locate the skill name in plan_str.
                position = -1
                if plan_str and raw_name:
                    # Find the skill name position in the string.
                    # Quoted forms are more accurate.
                    search_patterns = [
                        f'"{raw_name}"',  # "skill_name"
                        f"'{raw_name}'",  # 'skill_name'
                        raw_name  # skill_name (fallback)
                    ]
                    
                    for pattern in search_patterns:
                        start_pos, end_pos = find_text_position(plan_str, pattern)
                        if start_pos != -1:
                            # If a quoted form is found, exclude quote positions.
                            if pattern.startswith('"') or pattern.startswith("'"):
                                position = [start_pos + 1, end_pos - 1]
                            else:
                                position = [start_pos, end_pos]
                            break
                
                errors.append({
                    "message": f"Task {ti}: unknown skill '{raw_name}' not found in skill library",
                    "position": position,
                    "error_type": "invalid_skill"
                })

    return len(errors) == 0, errors


# =====================
# Edge Type and Node Content Consistency Validation/Fix
# =====================

def _extract_tbd_vars(task: Dict[str, Any]) -> Set[str]:
    """Extract all variable names from tbd:xxx placeholders in a task node."""
    task_str = json.dumps(task, ensure_ascii=False)
    return set(re.findall(r'tbd:(\w+)', task_str))


def _extract_produced_vars(task: Dict[str, Any]) -> Set[str]:
    """Extract variable names from the produces list in a task node."""
    produces = task.get("produces")
    if isinstance(produces, list):
        return {str(v) for v in produces if v}
    if isinstance(produces, str) and produces:
        return {produces}
    return set()


def _extract_condition_vars(condition: str) -> Set[str]:
    """Extract referenced variable names from a condition, such as 'cluster_location != null' -> {'cluster_location'}."""
    if not condition:
        return set()
    # Remove common comparison operators and values, then extract identifiers.
    # Conditions are usually formatted as: var_name != null / var_name == value.
    tokens = re.findall(r'[A-Za-z_]\w*', condition)
    # Exclude common keywords.
    keywords = {"null", "true", "false", "None", "True", "False", "and", "or", "not"}
    return {t for t in tokens if t not in keywords}


def fix_and_validate_edge_consistency(
    tasks: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    enable_fix: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate and fix consistency between edge types and node content.

    Rules:
    1. For a normal edge u->v, if v uses tbd:X and X is in u.produces,
       the edge should be conditional with condition X != null.
    2. For a conditional edge u->v, if the condition references no variable in u.produces
       and v does not use any u.produces variable as tbd, the edge should be normal.

    Args:
        tasks: Unified task list containing task_id, produces, required_skills, etc.
        edges: Parsed edge list in dictionary format with from, to, type, and condition.
        enable_fix: Whether automatic fixes are enabled.

    Returns:
        (whether all are consistent, error list)
    """
    errors: List[str] = []

    # Build task_id -> task node map.
    task_map: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        tid = t.get("task_id")
        if tid:
            task_map[tid] = t

    # Build task_id -> produces variable set.
    produces_map: Dict[str, Set[str]] = {}
    for tid, t in task_map.items():
        produces_map[tid] = _extract_produced_vars(t)

    # Build task_id -> tbd variable set.
    tbd_map: Dict[str, Set[str]] = {}
    for tid, t in task_map.items():
        tbd_map[tid] = _extract_tbd_vars(t)

    for edge in edges:
        u = edge.get("from", "")
        v = edge.get("to", "")
        etype = edge.get("type", "normal")
        condition = edge.get("condition")

        if not u or not v:
            continue
        if u not in task_map or v not in task_map:
            continue

        u_produces = produces_map.get(u, set())
        v_tbd = tbd_map.get(v, set())

        # Intersection between tbd variables used by v and variables produced by u.
        shared_vars = u_produces & v_tbd

        if etype == "normal":
            # Check: normal edge, but v uses tbd variables produced by u, so it should be conditional.
            if shared_vars:
                # Use the first shared variable to build the condition expression.
                cond_var = sorted(shared_vars)[0]
                cond_expr = f"{cond_var} != null"
                if enable_fix:
                    edge["type"] = "conditional"
                    edge["condition"] = cond_expr
                else:
                    errors.append({
                        "message": (
                            f"Edge {u}->{v} is normal, but {v} uses tbd variables "
                            f"{shared_vars} produced by {u}; it should be conditional"
                        ),
                        "error_type": "edge_type_mismatch"
                    })

        elif etype == "conditional":
            # Check: conditional edge, but the condition variable is not in u.produces
            # and v does not use any variable produced by u.
            cond_vars = _extract_condition_vars(condition) if condition else set()
            cond_refs_produce = bool(cond_vars & u_produces)
            v_uses_produce = bool(shared_vars)

            if not cond_refs_produce and not v_uses_produce:
                if enable_fix:
                    edge["type"] = "normal"
                    edge.pop("condition", None)
                else:
                    errors.append({
                        "message": (
                            f"Edge {u}->{v} is conditional with condition {condition}, "
                            f"but {u} does not produce any variable referenced by the condition, "
                            f"and {v} does not use any output from {u}; it should be normal"
                        ),
                        "error_type": "edge_type_mismatch"
                    })

    return len(errors) == 0, errors


# =====================
# Dependency Validation (Existence and Acyclicity)
# =====================

def validate_dependency_validity(tasks: List[Dict[str, Any]], plan_str: str = None) -> Tuple[bool, List[str]]:
    errors, task_ids = [], {t.get("task_id") for t in tasks if isinstance(t, dict)}
    for ti, task in enumerate(tasks):
        for dep in task.get("dependencies", []):
            if not isinstance(dep, str):
                errors.append(f"Task {ti}: dependency must be a string, but got {type(dep)}")
                continue
            if dep not in task_ids:
                # Try to locate the dependency ID in plan_str.
                position = -1
                if plan_str and dep:
                    # Find the dependency ID position in the string.
                    search_patterns = [
                        f'"{dep}"',  # "T999"
                        f"'{dep}'",  # 'T999'
                        dep  # T999 (fallback)
                    ]
                    
                    for pattern in search_patterns:
                        start_pos, end_pos = find_text_position(plan_str, pattern)
                        if start_pos != -1:
                            # If a quoted form is found, exclude quote positions.
                            if pattern.startswith('"') or pattern.startswith("'"):
                                position = [start_pos + 1, end_pos - 1]
                            else:
                                position = [start_pos, end_pos]
                            break
                
                errors.append({
                    "message": f"Task {ti}: dependent task '{dep}' does not exist",
                    "position": position,
                    "error_type": "dangling_dependency"
                })
    return len(errors) == 0, errors

def _has_cycle_from(node: str, graph: Dict[str, List[str]], color: Dict[str, int]) -> bool:
    """Depth-first cycle detection from a node using WHITE/GRAY/BLACK coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color[node] = GRAY
    for nxt in graph.get(node, []):
        if nxt not in color:
            continue
        if color[nxt] == GRAY:
            return True
        if color[nxt] == WHITE and _has_cycle_from(nxt, graph, color):
            return True
    color[node] = BLACK
    return False


def validate_acyclicity(tasks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    errors, graph = [], defaultdict(list)
    task_ids = {t.get("task_id") for t in tasks}
    for t in tasks:
        tid = t.get("task_id")
        for dep in t.get("dependencies", []):
            if isinstance(dep, str):
                graph[dep].append(tid)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {tid: WHITE for tid in task_ids}
    for tid in task_ids:
        if color[tid] == WHITE and _has_cycle_from(tid, graph, color):
            errors.append(f"Detected cyclic dependency involving task '{tid}'")
            break
    return len(errors) == 0, errors


# =====================
# Basic Schema Validation
# =====================

def validate_schema(tasks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Level 1 validation: check whether the structure meets requirements.
    """
    errors: List[str] = []
    required_keys = ["task_id", "required_skills", "dependencies"]

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"Task {i}: must be a dict")
            continue

        # Check whether required keys exist.
        for k in required_keys:
            if k not in task:
                errors.append(f"Task {i}: missing required key '{k}'")

        # Type validation.
        if "task_id" in task and not isinstance(task["task_id"], str):
            errors.append(f"Task {i}: 'task_id' must be a string")

        if "required_skills" in task and not isinstance(task["required_skills"], list):
            errors.append(f"Task {i}: 'required_skills' must be a list")

        if "dependencies" in task and not isinstance(task["dependencies"], list):
            errors.append(f"Task {i}: 'dependencies' must be a list")

    return len(errors) == 0, errors

def validate_meta_schema(meta_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the schema of the meta data block.
    """
    errors: List[str] = []
    
    # Support both description and reasoning keys.
    for key in ("description", "reasoning"):
        if key in meta_data and not isinstance(meta_data[key], str):
            errors.append(f"Key 'meta.{key}' must be a string.")

    if "shared_skill_groups" in meta_data:
        groups = meta_data["shared_skill_groups"]
        if not isinstance(groups, list):
            errors.append("Key 'meta.shared_skill_groups' must be a list.")
        else:
            for i, group in enumerate(groups):
                if not isinstance(group, list):
                    errors.append(f"Element {i} in 'shared_skill_groups' must be a list, but got {type(group).__name__}.")
                else:
                    for j, item in enumerate(group):
                        if not isinstance(item, str):
                            errors.append(f"Item {j} in group {i} of 'shared_skill_groups' must be a string.")
    
    return len(errors) == 0, errors


# =====================
# Public Main Flow
# =====================

def validate_complete(json_string: str, robot_skill_library: Dict[str, Any], enable_fix: bool = True) -> Dict[str, Any]:
    """
    Complete validation and automatic fix flow.
    """
    result: Dict[str, Any] = {
        "level1_format": {"valid": False, "errors": []},
        "level1_adapter": {"valid": False, "errors": []},
        "level1_meta_schema": {"valid": False, "errors": []},
        "level1_schema": {"valid": False, "errors": []},
        "level2_skills": {"valid": False, "errors": []},
        "level2_dependencies": {"valid": False, "errors": []},
        "level2_acyclicity": {"valid": False, "errors": []},
        "level2_edge_consistency": {"valid": False, "errors": []},
        "overall_valid": False,
        "fixed_data": None,
    }

    # Use the existing full static validation without automatic fixes.
    # 1. Parse with auto-fix.
    fmt_ok, fmt_err, data = parse_json_with_auto_fix(json_string)
    result["level1_format"]["valid"] = fmt_ok
    if not fmt_ok:
        result["level1_format"]["errors"] = [fmt_err]
        return result

    # 2. Adapt format to a unified task list.
    adapt_ok, adapt_err, universal_tasks, meta_data = adapt_plan_to_universal_format(data)
    result["level1_adapter"]["valid"] = adapt_ok
    if not adapt_ok:
        result["level1_adapter"]["errors"] = [adapt_err]
        return result
    
    meta_ok = True  # Default to True if there is no meta block.
    if meta_data is not None:
        meta_ok, meta_errs = validate_meta_schema(meta_data)
        result["level1_meta_schema"]["valid"] = meta_ok
        result["level1_meta_schema"]["errors"] = meta_errs
    else:
        result["level1_meta_schema"]["valid"] = True

    # 3. Fix and validate the unified format.
    if enable_fix:
        fix_schema_defaults(universal_tasks)
        fix_dependencies_type(universal_tasks)

    schema_ok, schema_errs = validate_schema(universal_tasks)
    result["level1_schema"]["valid"] = schema_ok
    result["level1_schema"]["errors"] = schema_errs
    if not schema_ok: return result

    skill_ok, skill_errs = fix_and_validate_skills(
        universal_tasks, robot_skill_library, json_string, enable_fix=enable_fix
    )
    result["level2_skills"]["valid"] = skill_ok
    result["level2_skills"]["errors"] = skill_errs

    dep_ok, dep_errs = validate_dependency_validity(universal_tasks, json_string)
    result["level2_dependencies"]["valid"] = dep_ok
    result["level2_dependencies"]["errors"] = dep_errs

    dag_ok, dag_errs = validate_acyclicity(universal_tasks)
    result["level2_acyclicity"]["valid"] = dag_ok
    result["level2_acyclicity"]["errors"] = dag_errs

    # Edge type and node content consistency validation, only for task_graph format.
    edge_ok = True
    if "task_graph" in data and isinstance(data["task_graph"], dict):
        graph_edges = data["task_graph"].get("edges", [])
        edge_ok, edge_errs = fix_and_validate_edge_consistency(
            universal_tasks, graph_edges, enable_fix=enable_fix
        )
        result["level2_edge_consistency"]["valid"] = edge_ok
        result["level2_edge_consistency"]["errors"] = edge_errs
    else:
        result["level2_edge_consistency"]["valid"] = True

    # 5. Summarize.
    is_valid = all([fmt_ok, adapt_ok, meta_ok, schema_ok, skill_ok, dep_ok, dag_ok, edge_ok])
    result["overall_valid"] = is_valid

    if is_valid:
        # Fill fixed_data only when valid and fixes are enabled.
        if enable_fix:
            # Write the fixed unified task list back to the original data structure.
            if "atomic_tasks" in data:
                data["atomic_tasks"] = universal_tasks
            elif "task_graph" in data:
                for task in universal_tasks:
                    task.pop("dependencies", None)
                data["task_graph"]["nodes"] = universal_tasks
            
            result["fixed_data"] = json.dumps(data, indent=2, ensure_ascii=False)

    return result
