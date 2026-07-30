feedback_plan_context: str = (
    """
### Real-time Feedback Context
{feedback_str}
""".strip()
)

############################## Main Prompt Template ##############################

TASK_PLAN_HEAD = """
{master_context}

### Current Available Robots
{available_robots}

{feedback_context_section}
### User Instruction
"{instruction}"

""".strip()

PHASE_TASK_PLAN_RESPONSE_FORMAT: str = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1.  Primary Goal: Decompose the `User Instruction` into a phase of fine-grained tasks, strictly adhering to the Parameterization Rules & Ontology.
2.  The Planning Boundary Rule (Important): Your plan phase MUST END before the first task that depends on unknown information.
3.  Planning Flow:
    - If `Real-time Feedback` exists: Your new plan must adapt to the feedback.
    - If it's an initial plan: If key information is missing, the plan's only goal is Information Gathering.
4. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
5. The environment model only reflects known information; unknown targets may require exploration.
6. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:2"` means 2 Humanoid executes navigate<Hotel-1>.
- `dependencies`: A flat list of predecessor task_ids, e.g. `["T1", "T2"]`.
- `shared_skill_groups`: Skills that must be assigned to the same robot, grouped by execution phase. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks).

### Result:
```json
{{
  "atomic_tasks": [
    {{
      "task_id": "T1",
      "location": "<The specific, standardized location_identifier>",
      "required_skills": ["robot_type:skill_str:robot_count", ...],
      "dependencies": []
    }}
  ],
  "meta": {{
    "reasoning": "Concise explanation of why this plan was generated and the key decisions.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }}
}}
""".strip()

FULL_TASK_PLAN_RESPONSE_FORMAT = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1. Generate a complete task graph as a JSON object, including both the `task_graph` and a `meta` summary.
2. If `Real-time Feedback` exists: Your new plan MUST adapt to the feedback.
3. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
4. The environment model only reflects known information; unknown targets may require exploration.
5. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:2"` means 2 Humanoid executes navigate<Hotel-1>.
- `edges`: Each edge is a string. Normal edge: `"T1->T2"`. Conditional edge: `"T1->T2:condition_expression"`. e.g. `"T3->T4:target_location != null"`.
- `shared_skill_groups`: Skills that must be assigned to the same robot, grouped by execution phase. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks).

### Result:
```json
{{
  "meta": {{
    "reasoning": "Concise explanation of why this plan was generated and the key decisions.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }},
  "task_graph": {{
    "nodes": [
      {{
        "task_id": "T1",
        "location": "<The specific, standardized location_identifier>",
        "required_skills": ["robot_type:skill_str:robot_count", ...],
        "produces": ["fact_name1", "fact_name2"]
      }},
      {{
        "task_id": "T2",
        "location": "tbd:location_identifier",
        ...
      }}
    ],
    "edges": [
      "T1->T2", "T3->T4:fact_name1 != null"
    ]
  }}
}}
```
""".strip()

PHASE_TASK_PLAN_TEMPLATE = f"{TASK_PLAN_HEAD}\n\n{PHASE_TASK_PLAN_RESPONSE_FORMAT}"
FULL_TASK_PLAN_TEMPLATE = f"{TASK_PLAN_HEAD}\n\n{FULL_TASK_PLAN_RESPONSE_FORMAT}"



######################################################################
############################## Initial Planning Template ##############################

INITIAL_TASK_PLAN_HEAD = """
{master_context}

### Current Available Robots
{available_robots}

### User Instruction
"{instruction}"

""".strip()

INITIAL_PHASE_TASK_PLAN_RESPONSE_FORMAT = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1. Primary Goal: Decompose the `User Instruction` into a phase of fine-grained tasks, strictly adhering to the Parameterization Rules & Ontology.
2. The Planning Boundary Rule (Important): Your plan phase MUST END before the first task that depends on unknown information.
3. If key information is missing, the plan's only goal is Information Gathering.
4. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
5. The environment model only reflects known information; unknown targets may require exploration.
6. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.
7. The output format means you cannot plan at the level of individual robots (including reasoning) - only at the robot type level.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:2"` means 2 Humanoid executes navigate<Hotel-1>.
- `dependencies`: A flat list of predecessor task_ids, e.g. `["T1", "T2"]`.
- `shared_skill_groups`: Skills that must be assigned to the same robot, grouped by execution phase. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks).

### Result:
```json
{{
  "atomic_tasks": [
    {{
      "task_id": "T1",
      "location": "<The specific, standardized location_identifier>",
      "required_skills": ["robot_type:skill_str:robot_count", ...],
      "dependencies": []
    }}
  ],
  "meta": {{
    "reasoning": "Concise explanation of why this plan was generated and the key decisions.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }}
}}
""".strip()

INITIAL_FULL_TASK_PLAN_RESPONSE_FORMAT = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1. Generate a complete task graph to fulfill the user's instruction, focusing on the "Master Context" section.
2. You output must strictly follow the "Output Format" and "Graph Conventions" sections.
3. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
4. The environment model only reflects known information; unknown targets may require exploration.
5. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.
6. The output format means you cannot plan at the level of individual robots (including reasoning) - only at the robot type level.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:2"`, `UAV:navigate<tbd:target_location>:1`.
- `edges`: Each edge is a compact string. Normal edge: `"T1->T2"`. Conditional edge: `"T1->T2:condition_expression"`. e.g. `"T3->T4:target_location != null"`.
- `shared_skill_groups`: Skills that must be assigned to the same robot. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks). 

### Result:
```json
{{
  "meta": {{
    "reasoning": "Concise explanation of why this plan was generated and the key decisions.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }},
  "task_graph": {{
    "nodes": [
      {{
        "task_id": "T1",
        "location": "<The specific, standardized location_identifier>",
        "required_skills": ["robot_type:skill_str:robot_count", ...],
        "produces": ["fact_identifier", ...]
      }},
      {{
        "task_id": "T2",
        "location": "tbd:fact_identifier",
        "required_skills": ...,
      }}
    ],
    "edges": [
      "T1->T2:fact_identifier != null"
    ]
  }}
}}
```
""".strip()

INITIAL_PHASE_TASK_PLAN_TEMPLATE = f"{INITIAL_TASK_PLAN_HEAD}\n\n{INITIAL_PHASE_TASK_PLAN_RESPONSE_FORMAT}"
INITIAL_FULL_TASK_PLAN_TEMPLATE = f"{INITIAL_TASK_PLAN_HEAD}\n\n{INITIAL_FULL_TASK_PLAN_RESPONSE_FORMAT}"


######################################################################
############################## Replanning Template ##############################

REPLANNING_TASK_PLAN_HEAD = """
{master_context}

{feedback_context_section}

### Previous Plan
{previous_plan}

### User Instruction
"{instruction}"

""".strip()

REPLANNING_PHASE_TASK_PLAN_RESPONSE_FORMAT = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1. If the feedback event type is EVALUATION_TRIGGER, generate ONLY the next-step plan that continues from the previous plan. If you think the task is complete, output an empty atomic_tasks ("atomic_tasks": []):
2. For any other feedback event type, update or repair the previous plan according to the feedback. Critically, do NOT output any atomic tasks that have already been completed; only output the remaining and any newly added or modified atomic tasks.
3. The "task_id" in shared_skill_groups and atomic_tasks must restart from T1 and correspond correctly.
4. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
5. The environment model only reflects known information; unknown targets may require exploration.
6. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.
7. The output format means you cannot plan at the level of individual robots (including reasoning) - only at the robot type level.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:1"`.
- `dependencies`: A flat list of predecessor task_ids, e.g. `["T1", "T2"]`.
- `shared_skill_groups`: Skills that must be assigned to the same robot, grouped by execution phase. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks).

### Result:
```json
{{
  "atomic_tasks": [
    {{
      "task_id": "T1",
      "location": "<The specific, standardized location_identifier>",
      "required_skills": ["robot_type:skill_str:robot_count", ...],
      "dependencies": []
    }}
  ],
  "meta": {{
    "reasoning": "Explain why this updated plan was generated and the key decisions made.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }}
}}
""".strip()

REPLANNING_FULL_TASK_PLAN_RESPONSE_FORMAT = """
## Core Directives & Planning Logic
You are an expert multi-robot task planner. Follow these rules strictly:
1. Adjust the plan based on the "### Real-time Feedback Context" and "### Previous Plan." If all skills succeed, output the next-step plan to complete the user instruction.
2. If failure or an unexpected event occurs, refer to the "## Master Context ##" section for careful consideration and modify the Previous Plan accordingly.
3. If you think the task is complete, output an empty task graph ("nodes": [], "edges": []).
4. Never re-outputting already completed atomic tasks.
5. The "task_id" in shared_skill_groups and task_graph must restart from T1 and correspond correctly.
6. When a task is parallelizable and benefits from multiple robots, use multiple robots rather than defaulting to one.
7. The environment model only reflects known information; unknown targets may require exploration.
8. Transporting objects requires heterogeneous collaboration, with careful attention to the temporal ordering of skill execution.
9. The output format means you cannot plan at the level of individual robots (including reasoning) - only at the robot type level.

### Output Format
- `required_skills`: Each skill is a string `robot_type:skill_str:robot_count`. e.g. `"Humanoid:navigate<Hotel-1>:1"`, `"UAV:navigate<tbd:target_location>:1"`.
- `edges`: Each edge is a compact string. Normal edge: `"T1->T2"`. Conditional edge: `"T1->T2:condition_expression"`. e.g. `"T3->T4:target_location != null"`.
- `shared_skill_groups`: Skills that must be assigned to the same robot, grouped by execution phase. Each skill T<task_id>.<skill_index>" must come from tasks in the same phase, where <skill_index> is the 0-based index of that skill in the task's required_skills list. All tasks within a single group MUST be parameter-ready at the same time (do not mix concrete and 'tbd' tasks).

### Result:
```json
{{
  "meta": {{
    "reasoning": "Concise explanation of why this plan was generated and the key decisions.",
    "shared_skill_groups": [
      ["T1.0", "T2.1", ...]
    ]
  }},
  "task_graph": {{
    "nodes": [
      {{
        "task_id": "T1",
        "location": "<The specific, standardized location_identifier>",
        "required_skills": ["robot_type:skill_str:robot_count", ...],
        "produces": ["fact_name1", "fact_name2"]
      }}
    ],
    "edges": [
      "T1->T2:fact_identifier != null"
    ]
  }}
}}
```
""".strip()

REPLANNING_PHASE_TASK_PLAN_TEMPLATE = f"{REPLANNING_TASK_PLAN_HEAD}\n\n{REPLANNING_PHASE_TASK_PLAN_RESPONSE_FORMAT}"
REPLANNING_FULL_TASK_PLAN_TEMPLATE = f"{REPLANNING_TASK_PLAN_HEAD}\n\n{REPLANNING_FULL_TASK_PLAN_RESPONSE_FORMAT}"
