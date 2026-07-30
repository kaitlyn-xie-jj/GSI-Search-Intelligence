# ============================================================================
# Shared core definition blocks, split into standalone snippets for on-demand assembly.
# ============================================================================

ATOMIC_TASK_DEFINITION = """
- Atomic Task: 
  - Each atomic task MUST have a unique task_id (e.g., "T1", "T2").
""".strip()

TASK_LOCATION_RULES = """
- Task Location Rules: 
  - For movement/transport tasks, `location` is the DESTINATION. (e.g., For Task: "Carry cargo to hospital", the location is hospital).
  - For search tasks, `location` is the AREA or TARGET being observed.
  - If a search skill is required but no explicit area is given, infer a region from the environment model; if no model exists, set the location/area to `cybertown`.
  - For other cases where no specific destination applies, use `current_location`, e.g. take off.
""".strip()

TASK_NATURE_PRINCIPLE = """
- Task Nature Principle:
  - **Information-Gathering Tasks (IGTs)**: These are tasks whose primary purpose is to collect information about the environment's state. Their output is data (e.g., locations of objects).
  - **Execution Tasks (ETs)**: These are tasks that change the state of the environment based on known, concrete parameters. Their purpose is to act, not to learn. Skills like `load`, `navigate` are examples.
""".strip()

PARAMETERIZATION_RULES = """
- Parameterization Rules & Ontology:
  - Location Identifiers:
  -   **Named Area**: Use name. (e.g., `Street Segment-33`, `campus-1`)
  -   **Point Radius**: Use `PointRadius_<Point>_<Radius>m`. (e.g., `PointRadius_MarkedPoint_200m`)
  -   **Boundary**: Use `BoundarySelection_MarkedArea` for user-marked areas.
  - Target Ontology:
  -   **Event**: such as `illegal_parking`,`traffic_violation`,`crowd_gathering`
  -   **Object**: such as `Suspicious_Person`,`Blue_SUV`, `Blue_SUV_with_Fire`,`Person_WhiteClothes_with_Camera`, `Fire_Spot`, `Equipment_Hazard`, `Hazardous_Leak`, `charging_dock`
  - Rule for selection:
    - If the goal is to detect the act or occurrence, use an **Event** target.
    - If the goal is to find a specific physical thing, use an **Object** target.
""".strip()

ADDITIONAL_NOTES = """
""".strip()

core_definitions_replanning = """
  - If you determine that no new plan should be generated, you may return an empty list.
""".strip()

graph_conventions_full = """
- A Directed Acyclic Graph. Nodes are atomic tasks. Edges define control flow and data flow.
- Nodes: Must contain `task_id`, `location`, `required_skills`. Can optionally `produces` facts, e.g. "produces": ["target_location", "target_number"]
- Edges:
  - `type: "normal"`: Sequential execution.
  - `type: "conditional"`: Branching. Requires a `condition` field.
- If a parameter depends on the result of a previous task, use "tbd:xxx" as a placeholder where xxx come from the produces of a previous task..
- "tbd" and "produce" usually appear together or not at all; one cannot appear without the other.
""".strip()


def build_core_definitions(goal_type_notes: str = "") -> str:
    """Assemble the complete core_definitions text for initial planning."""
    parts = [
        ATOMIC_TASK_DEFINITION,
        TASK_LOCATION_RULES,
        PARAMETERIZATION_RULES,
        # ADDITIONAL_NOTES,
    ]
    if goal_type_notes:
        parts.append(f"  {goal_type_notes}")
    return "\n".join(parts)


def build_core_definitions_replanning(goal_type_notes: str = "") -> str:
    """Assemble the complete core_definitions_replanning text for replanning."""
    parts = [
        PARAMETERIZATION_RULES,
        # ADDITIONAL_NOTES,
    ]
    if goal_type_notes:
        parts.append(f"  {goal_type_notes}")
    return "\n".join(parts)



################################## Approach 1: Generate Near-Term Plan ##################################

master_text: str = (
    """
## Master Context ##
### 1. Environment Model
{env_description}

### 2. Robot Skill Library
{skill_set_markdown}
# Plans MUST be grounded in this library. All parameters in <...> MUST follow the naming rules.

### 3. Core Definitions
{core_definitions}
""".strip()
)

master_text_no_env: str = (
    """
## Master Context ##
### 1. Robot Skill Library
{skill_set_markdown}
# Plans MUST be grounded in this library. All parameters in <...> MUST follow the naming rules.

### 2. Core Definitions
{core_definitions}
""".strip()
)

################################## Approach 2: Generate Full Plan ##################################

master_text_full = """
## Master Context ##
### 1. Environment Model
{env_description}

### 2. Robot Skill Library
{skill_set_markdown}
# Plans MUST be grounded in this library. All parameters in <...> MUST follow the naming rules.

### 3. Core Definitions
{core_definitions}

### 4. Graph Conventions
{graph_conventions_full}
""".strip()

master_text_full_no_env = """
## Master Context ##
### 1. Robot Skill Library
{skill_set_markdown}
# Plans MUST be grounded in this library. All parameters in <...> MUST follow the naming rules.

### 2. Core Definitions
{core_definitions}

### 3. Graph Conventions
{graph_conventions_full}
""".strip()
