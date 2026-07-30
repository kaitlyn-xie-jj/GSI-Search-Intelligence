from typing import Dict, Any, List
import re

from modules.task_solver.sgi_planner.utils import get_robot_skills
from modules.config.system_config import config

SKILL_SCHEMAS = {
    # Skill name: { 'pattern': regex, 'params': [semantic parameter names] }
    
    # UAV/FW_UAV Skills
    "take_off": {
        "pattern": re.compile(r"take_off$"),
        "params": []
    },
    "return_home": {
        "pattern": re.compile(r"return_home$"),
        "params": []
    },
    "search": {
        "pattern": re.compile(r"search<([^>]+)>_for<([^>]+)>$"),
        "params": ["area", "target"]
    },
    "broadcast": {
        "pattern": re.compile(r"broadcast<([^>]+)>$"),
        "params": ["target"] 
    },
    "handle_hazard": { 
        "pattern": re.compile(r"handle_hazard<([^>]+)>$"),
        "params": ["target"]
    },

    # Quadruped Skills
    "guide": { 
        "pattern": re.compile(r"guide<([^>]+)>_to<([^>]+)>$"),
        "params": ["target", "location"]
    },
    
    # Humanoid Skills
    "place": {
        "pattern": re.compile(r"place<([^>]+)>_on<([^>]+)>$"),
        "params": ["object1", "object2"]
    },

    # Shared Skills (with different robot implementations)
    "navigate": {
        "pattern": re.compile(r"navigate<([^>]+)>$"),
        "params": ["location"]  # 'object1' -> 'location'
    },
    "take_photo": {
        "pattern": re.compile(r"take_photo<([^>]+)>$"),
        "params": ["target"]  # 'object1' -> 'target'
    },
    "follow": {
        "pattern": re.compile(r"follow<([^>]+)>$"),
        "params": ["target"]
    },

    "sync_wait": {
        "pattern": re.compile(r"sync_wait$"),
        "params": []
    }
}

robot_skill_source =  {
    "UAV": {
        "skill_1": {
            "name": "take_off",
            "description": "UAV needs to take off before flying.",
            "precondition": "The UAV is on a stable, flat surface with clear vertical space. System status is nominal.",
            "effect": "The UAV is airborne and hovering, ready for commands."
        },
        "skill_2": {
            "name": "return_home",
            "description": "",
            "precondition": "A clear flight path to the home base exists and its landing pad is clear.",
            "effect": "The UAV is safely landed at its home base and enters standby/charging mode."
        },
        "skill_3": {
            "name": "navigate<location>",
            "description": "Moves to <location>.",
            "precondition": "A reasonably clear flight path exists.",
            "effect": "The UAV's position is updated to the specified <location>."
        },
        "skill_4": {
            "name": "follow<target>",
            "description": "Aerially follows a moving <target>, maintaining a set distance and line-of-sight.",
            "precondition": "The <target> is identified and within the UAV's sensor range.",
            "effect": "The UAV's movement is synchronized with the <target>'s."
        },
        "skill_5": {
            "name": "take_photo<area_or_target>",
            "description": "Captures a photo of the specified <area> or <target> for data/evidence collection. The robot must be at the <target> location first (use navigate if needed).",
            "precondition": "The UAV is at a operational altitude and its camera is functional.",
            "effect": "A wide-field-of-view digital image is created and stored with metadata."
        },
        "skill_6": {
            "name": "broadcast<target>",
            "description": "Broadcasts voice to a specified <target> for hailing, warning, or dispersal. The robot must be near the <target> first (use navigate if needed).",
            "precondition": "The robot is within audible range; the loudspeaker is functional and local regulations are respected.",
            "effect": "A clear voice message is broadcast; nearby persons are alerted/warned and the event is logged."
        },
        "skill_7": { 
            "name": "handle_hazard<target>",
            "description": "Handle hazardous leaks/equipment faults/fires for <target>. The robot must be at the <target> location first (use navigate if needed).",
            "precondition": "Hazard identified; proper tools available and regulations followed.",
            "effect": "Hazard is contained or neutralized; status logged."
        },
        "skill_8": {
            "name": "search<area>_for<target>",
            "description": "Search or patrol a defined <area> from the air to locate a <target> whose location is uncertain, requiring clear, well-lit visibility. Typically produces location info of the <target>.",
            "precondition": "The boundaries of the search <area> are defined.",
            "effect": "The precise location of the <target> is logged if found; otherwise, the <area> is marked 'target not found'."
        },
        "skill_9": {
            "name": "sync_wait",
            "description": "Hover in place and wait for synchronization with other robots.",
            "precondition": "The UAV is airborne and stable.",
            "effect": "The UAV maintains its position and is ready to coordinate with other robots."
        }
    },
    # "FW_UAV": {
    #     "skill_1": {
    #         "name": "take_off",
    #         "description": "",  
    #         "precondition": "Runway/launch method available; system nominal.",
    #         "effect": "FW UAV is airborne and ready."
    #     },
    #     "skill_2": {
    #         "name": "return_home",
    #         "description": "",  
    #         "precondition": "Home path and landing site clear.",
    #         "effect": "FW UAV lands at home and enters standby."
    #     },
    #     "skill_3": {
    #         "name": "navigate<location>",
    #         "description": "Moves to <location>.",
    #         "precondition": "A safe flight path exists.",
    #         "effect": "Position updated to <location>."
    #     },
    #     "skill_4": {
    #         "name": "search<area>_for<target>",
    #         "description": "Search or patrol a defined <area> from the air to locate a <target>, requiring clear, well-lit visibility.",
    #         "precondition": "The boundaries of the search <area> are defined.",
    #         "effect": "The precise location of the <target> is logged if found; otherwise, the <area> is marked 'target not found'."
    #     },
    #     "skill_5": {
    #         "name": "sync_wait",
    #         "description": "Hover in place and wait for synchronization with other robots.",
    #         "precondition": "The FW UAV is airborne and stable.",
    #         "effect": "The FW UAV maintains its position and is ready to coordinate with other robots."
    #     }
    # },
    "UGV": {
        "skill_1": {
            "name": "navigate<location>",
            "description": "Drives to a specified <location> with or without carried cargo/injured person/assembly component.",
            "precondition": "A navigable ground path exists between the current position and the target <location>.",
            "effect": "The vehicle's position is updated to the target <location>."
        },
        "skill_2": {
            "name": "broadcast<target>",
            "description": "Broadcasts voice to a specified <target>. The robot must be near the <target> first (use navigate if needed).",
            "precondition": "The robot is within audible range; the loudspeaker is functional and local regulations are respected.",
            "effect": "A clear voice message is broadcast; nearby persons are alerted/warned and the event is logged."
        },
        "skill_3": {
            "name": "sync_wait",
            "description": "Wait for synchronization with other robots.",
            "precondition": "The UGV is stationary and stable.",
            "effect": "The UGV maintains its position and is ready to coordinate with other robots."
        }
    },
    "Quadruped": {
        "skill_1": {
            "name": "navigate<location>",
            "description": "Moves to <location>.",
            "precondition": "A path exists, which may be non-flat or unstructured.",
            "effect": "The robot's position is updated to the <location>."
        },
        "skill_2": {
            "name": "follow<target>",
            "description": "Follows a moving <target>, maintaining a set distance and line-of-sight.",
            "precondition": "The <target> is identified and within the robot's sensor range.",
            "effect": "The robot's movement is synchronized with the <target>'s, and its location is broadcast to the team."
        },
        "skill_3": {
            "name": "take_photo<target>",
            "description": "Captures a detailed, ground-level photo of a specific <target>, with optional infrared capability. The robot must be at the <target> location first (use navigate if needed).",
            "precondition": "The robot has a clear line-of-sight to the <target>.",
            "effect": "A high-resolution digital image of the <target> is created and stored."
        },
        "skill_4": {
            "name": "search<area>_for<target>",
            "description": "Conducts a ground-level patrol search of <area> to locate a <target> whose location is uncertain, with optional infrared capability. Typically produces location info of the <target>.",
            "precondition": "A feasible patrol path exists; stability and terrain constraints are within limits.",
            "effect": "The quadruped completes cyclic patrol passes; anomalies are logged and can notify the team."
        },
        "skill_5": {
            "name": "guide<target>_to<location>",
            "description": "Guides a specified <target> to <location>.",
            "precondition": "Target can follow; route is feasible and safe.",
            "effect": "<Target> arrives at <location>; event logged."
        },
        "skill_6": {
            "name": "broadcast<target>",
            "description": "Broadcasts voice to a specified <target>. The robot must be near the <target> first (use navigate if needed).",
            "precondition": "The robot is within audible range; the loudspeaker is functional and local regulations are respected.",
            "effect": "A clear voice message is broadcast; nearby persons are alerted/warned and the event is logged."
        },
        "skill_7": {
            "name": "sync_wait",
            "description": "Wait for synchronization with other robots.",
            "precondition": "The quadruped is stationary and stable.",
            "effect": "The quadruped maintains its position and is ready to coordinate with other robots."
        }
    },
    "Humanoid": {
        "skill_1": {
            "name": "navigate<location>",
            "description": "Moves to <location>.",
            "precondition": "A clear, walkable path exists to the <location>.",
            "effect": "The robot's position is updated to the specified <location>."
        },
        "skill_2": {
            "name": "place<object1>_on<object2>",
            "description": "Places <object1> on <object2> (stack/mount/load). The robot must be at the objects' location first (use navigate if needed). For transport, use paired place<xx>_on<UGV> and place<xx>_on<ground> for loading and unloading.",
            "precondition": "Both targets accessible; placement is stable and safe.",
            "effect": "<object1> is stably placed on <object2>; status logged."
        },
        "skill_3": {
            "name": "sync_wait",
            "description": "Wait for synchronization with other robots.",
            "precondition": "The humanoid is stationary and stable.",
            "effect": "The humanoid maintains its position and is ready to coordinate with other robots."
        },
    }
}

robot_type_list = config.default_robot_types if config.default_robot_types else ["UAV", "FW_UAV", "UGV", "Quadruped", "Humanoid"]
robot_skill_library = get_robot_skills(robot_type_list, robot_skill_source)
