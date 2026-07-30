# modules/utils/var_dump.py
from __future__ import annotations
import json, os, threading
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

__all__ = [
    "set_default_dump_dir",
    "get_dumper",
    "dump_var",
    "VarDumper",
]

# --- helpers -----------------------------------------------------------------

def _json_default(o: Any):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    # Last resort: string repr (keeps dumper robust)
    return repr(o)

# --- per-process dumper ------------------------------------------------------

class VarDumper:
    """
    Minimal JSONL dumper.
    - One file per *process* (PID) => no cross-process contention.
    - Thread-safe inside the process.
    """
    def __init__(self, base_dir: Path, file_prefix: str = "vars") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.path = self.base_dir / f"temp_{file_prefix}.jsonl"
        self._lock = threading.Lock()

    def save(
        self,
        name: str,
        value: Any,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        rec = {
            # "ts": datetime.utcnow().isoformat() + "Z",
            # "pid": self.pid,
            "name": name,
            "value": value,
        }
        if meta:
            rec["meta"] = meta
        line = json.dumps(rec, ensure_ascii=False, default=_json_default)

        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return self.path

# --- global accessors --------------------------------------------------------

# Default dir can be set once per process. Falls back to:
#   (env VAR_DUMP_DIR) or "./var_dumps"
_DEFAULT_DIR: Optional[Path] = None
_DUMPER: Optional[VarDumper] = None

def set_default_dump_dir(base_dir: Path | str) -> None:
    """Set the per-process default directory where dumps go."""
    global _DEFAULT_DIR, _DUMPER
    _DEFAULT_DIR = Path(base_dir)
    _DUMPER = None  # recreate with the new dir on next get

def get_dumper(base_dir: Optional[Path | str] = None) -> VarDumper:
    """Get/create the per-process VarDumper singleton."""
    global _DEFAULT_DIR, _DUMPER
    if base_dir is not None:
        set_default_dump_dir(base_dir)
    if _DUMPER is None:
        base = (
            Path(_DEFAULT_DIR)
            if _DEFAULT_DIR is not None
            else Path(os.getenv("VAR_DUMP_DIR", "./var_dumps"))
        )
        _DUMPER = VarDumper(base)
    return _DUMPER

def dump_var(name: str, value: Any, *, meta: Optional[Dict[str, Any]] = None) -> Path:
    """
    Convenience function: dump immediately using the per-process singleton.
    Usage: dump_var("current_plan", plan, meta={"stage":"planning"})
    """
    return get_dumper().save(name, value, meta=meta)
