
import os, random, json
from typing import Dict, Optional
from pathlib import Path
from modules.config import (
    Category, AREA_TEMPLATES
)

REQUIRED_AREAS = {"water_body", "garden", "neighborhood"}
AREA_TYPES_POOL = set(AREA_TEMPLATES.keys())

def set_global_seed(seed: Optional[int]):
    """
    Set seeds for Python, NumPy (if present), and PyTorch (if present).
    Also fixes Python hash seed for reproducibility of dict iteration orders.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def rand_areas_per_district(district_layout: Dict[str, Dict], rng: random.Random) -> Dict[str, Dict[str, int]]:
    """
    Per district:
      - Must include: water_body / garden / neighborhood
      - Other types randomly selected (may be 0), each type count in [1,5]
    """
    plan = {}
    for d_id in district_layout.keys():
        # Place required 3 types first
        counts = {t: rng.randint(1, 5) for t in REQUIRED_AREAS}
        # Consider extra types: random subset
        extras = sorted(list(AREA_TYPES_POOL - REQUIRED_AREAS))
        rng.shuffle(extras)
        # Include each extra type with 0.5 probability
        for t in extras:
            if rng.random() < 0.5:
                counts[t] = rng.randint(1, 5)
        plan[d_id] = counts
    return plan

def rand_building_plan(tmpl_mgr, rng: random.Random) -> Dict[str, int]:
    """
    Buildings:
      - robot_base exactly 1
      - library in [1,10]
      - Other buildings (present in template library) in [0,3]
    """
    lib = tmpl_mgr.libs.get(Category.BUILDING.value, {}) or {}
    types = set(lib.keys())

    plan = {"robot_base": 1}
    if "library" in types:
        plan["library"] = rng.randint(1, 10)

    # Other buildings (excluding robot_base/library)
    for t in sorted(types - {"robot_base", "library"}):
        plan[t] = rng.randint(0, 3)
    return plan

def parse_robot_plan_arg(arg: Optional[str]) -> Optional[Dict[str, int]]:
    """
    Supports two forms:
      --robot-plan '{"UAV":3,"UGV":1}'
      --robot-plan @path/to/robots.json
    """
    if not arg:
        return None
    arg = arg.strip()
    if arg.startswith("@"):
        path = arg[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(arg)

def rand_prop_plan(tmpl_mgr, rng: random.Random) -> Dict[str, int]:
    """
    Props:
      - person / vehicle / cargo must exist, count >=3 and <=15
      - Other three types (equipment_failure / security_breach / boat) count in [0,5]
    """
    # Can filter by template library: tmpl_mgr.libs.get(Category.PROP.value, {})
    plan = {}
    # Required three types
    for must in ("person", "vehicle", "cargo"):
        plan[must] = rng.randint(3, 15)

    # Other three types (extend based on actual prop set as needed)
    for opt in ("equipment_failure", "security_breach", "boat"):
        plan[opt] = rng.randint(0, 5)
    return plan

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)