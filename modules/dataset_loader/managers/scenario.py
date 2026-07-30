import json
from pathlib import Path
from typing import Any, Dict
from functools import lru_cache


class ScenarioManager:
    """
    Scenario Manager

    Responsibilities:
    1. Manage locally cached scenario file paths
    2. Read JSON files on demand (scene_graph.json, plans.json)
    3. Use LRU cache to avoid repeated IO
    """

    def __init__(self, local_root: Path, type_name: str):
        self.root = local_root / "scenarios" / type_name
        self.type_name = type_name
        # Bind cached read function
        self.get_config = lru_cache(maxsize=128)(self._read_json)

    def _read_json(self, relative_path: str) -> Dict[str, Any]:
        """Internal read function"""
        path = self.root / relative_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Failed to read scenario file {path}: {e}")
        return {}

    def get_scene_graph(self, scenario_id: str) -> Dict[str, Any]:
        """Get scene graph data"""
        return self.get_config(f"{scenario_id}/scene_graph.json")

    def get_plans(self, scenario_id: str) -> Dict[str, Any]:
        """Get scenario statistics/planning data"""
        return self.get_config(f"{scenario_id}/plans.json")
