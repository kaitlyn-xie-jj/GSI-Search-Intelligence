import re
from typing import Dict, Any, List

def to_concise_robot_info(detailed_output: Dict[str, Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Convert detailed robot availability info into concise format.

    Args:
        detailed_output: e.g.,
            {
                "UAV": {"labels": ["UAV-1", "UAV-2"], "num": 2},
                "Quadruped": {"labels": ["Quadruped-3"], "num": 1}
            }

    Returns:
        Concise version, e.g.:
            {"UAV": [1, 2], "Quadruped": [3]}
    """
    id_tail_re = re.compile(r"(\d+)$")
    concise_output: Dict[str, List[int]] = {}

    for rtype, info in detailed_output.items():
        labels = info.get("labels", [])
        nums = []
        for label in labels:
            m = id_tail_re.search(label.strip())
            if m:
                nums.append(int(m.group(1)))
        if nums:
            concise_output[rtype] = nums

    return concise_output
