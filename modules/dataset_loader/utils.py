import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from datasets import load_dataset, Dataset
except ImportError:
    Dataset = None

# ==========================================
# JSONL Index & Fast Read Tools
# ==========================================


def build_jsonl_index(
    file_path: Path, id_column: str, cache: Optional[Dict[str, Dict[str, int]]] = None
) -> Dict[str, int]:
    """
    Build JSONL file index: {id: byte_offset}
    Prioritize reading .index file, otherwise scan in real-time.

    Args:
        file_path: JSONL file path
        id_column: ID field name
        cache: (optional) External cache dict to avoid repeated indexing
    """
    path_str = str(file_path)

    # 1. Check memory cache
    if cache is not None and path_str in cache:
        return cache[path_str]

    if not file_path.exists():
        return {}

    # 2. Check pre-built index file on disk (.index)
    index_path = file_path.with_name(file_path.name + ".index")
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
                if cache is not None:
                    cache[path_str] = index
                return index
        except Exception:
            pass  # Read failed, fallback to scan

    # 3. Real-time scan to build index
    print(f"Indexing file: {file_path.name}...")
    index = {}
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                # Only parse necessary ID field
                record = json.loads(line)
                record_id = str(record.get(id_column))
                if record_id:
                    index[record_id] = offset
            except json.JSONDecodeError:
                continue

    # 4. Update cache
    if cache is not None:
        cache[path_str] = index

    return index


def read_jsonl_record(
    file_path: Path, record_id: str, index_map: Dict[str, int]
) -> Optional[Dict[str, Any]]:
    """
    O(1) complexity read single record
    """
    if not file_path.exists():
        return None

    offset = index_map.get(str(record_id))
    if offset is None:
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        line = f.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None


# ==========================================
# Data Cleaning & Metadata Tools
# ==========================================


def normalize_goal_data(goal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize goal data: parse JSON string fields, supplement description"""
    data = dict(goal_data)  # Create copy

    # Parse nested JSON strings
    for field in ["goal_details", "meta"]:
        if field in data and isinstance(data[field], str):
            try:
                data[field] = json.loads(data[field])
            except json.JSONDecodeError:
                pass

    # Supplement description
    if "instruction" in data and "goal_details" in data:
        if isinstance(data["goal_details"], dict):
            data["goal_details"].setdefault("description", data.get("instruction", ""))
    return data


def parse_level_value(level_str: Any) -> int:
    """
    Parse "L1", "L2" strings to integers 1, 2.
    Returns -1 if cannot parse.
    """
    if isinstance(level_str, int):
        return level_str
    if isinstance(level_str, str) and level_str.upper().startswith("L"):
        try:
            return int(level_str[1:])
        except ValueError:
            pass
    return -1


def extract_goal_metadata(goal_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata for filtering from complete Goal record (lightweight)
    """
    # Ensure data is normalized (parse nested JSON)
    data = normalize_goal_data(goal_data)

    goal_details = data.get("goal_details", {})
    meta = data.get("meta", {})

    # Parse level lists (e.g., ["L0", "L1"])
    plan_levels = meta.get("plan_level", [])
    if not isinstance(plan_levels, list):
        plan_levels = [plan_levels] if plan_levels else []

    coor_levels = meta.get("coor_level", [])
    if not isinstance(coor_levels, list):
        coor_levels = [coor_levels] if coor_levels else []

    # Extract key fields
    return {
        "goal_id": goal_details.get("goal_id"),
        "goal_type": goal_details.get("goal_type", "unknown"),
        "language_level": meta.get("language_level", "L0"),
        "plan_level": plan_levels,
        "coor_level": coor_levels,
        # Helper fields: compute max Level value for numeric comparisons like < L3
        "max_plan_level": max([parse_level_value(l) for l in plan_levels] + [-1]),
        "max_coor_level": max([parse_level_value(l) for l in coor_levels] + [-1]),
        "max_language_level": parse_level_value(meta.get("language_level", "L0")),
    }


def detect_key_column(dataset: Dataset, candidate_columns: List[str]) -> Optional[str]:
    """Auto-detect available primary key column in dataset"""
    if not dataset:
        return None
    column_names = set(dataset.column_names)
    for col in candidate_columns:
        if col in column_names:
            return col
    return None


def load_dataset_from_file(file_path: Path) -> Dataset:
    """Load HF Dataset from file"""
    try:
        return load_dataset("json", data_files=str(file_path), split="train")
    except Exception as e:
        raise Exception(f"Failed to load dataset from file:{file_path}:{e}")
