from typing import Dict, Any, Optional, List

class UnifiedTemplateManager:
    """Unified template reader: provides the minimal API needed for scene construction.
    
    Only performs reads and lightweight validation; does not handle generation/placement logic.
    """

    def __init__(self, libs: Dict[str, Dict[str, Any]]):
        # Expected structure: { category(str) -> { type_key(str) -> template(dict) } }
        self.libs: Dict[str, Dict[str, Any]] = libs

    # Basic retrieval
    def categories(self) -> List[str]:
        return list(self.libs.keys())

    def has(self, category: str, type_key: str) -> bool:
        return category in self.libs and type_key in self.libs[category]

    def get_template(self, category: str, type_key: str) -> Optional[Dict[str, Any]]:
        lib = self.libs.get(category)
        if not lib:
            return None
        return lib.get(type_key)

    def get_all_types(self, category: str) -> List[str]:
        return list(self.libs.get(category, {}).keys())

    # Shortcut accessors for common fields
    def get_default_status(self, category: str, type_key: str) -> Optional[str]:
        tpl = self.get_template(category, type_key)
        if not tpl:
            return None
        st = tpl.get("status") or {}
        return st.get("default")

    def get_status_options(self, category: str, type_key: str) -> List[str]:
        tpl = self.get_template(category, type_key)
        if not tpl:
            return []
        st = tpl.get("status") or {}
        return list(st.get("options", []))

    def get_size(self, category: str, type_key: str, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tpl = self.get_template(category, type_key) or {}
        size = tpl.get("size")
        if size:
            return size
        return fallback or {"width": 1.0, "length": 1.0, "height": 1.0, "unit": "meters"}

    # Shortcut: safely read color/appearance (if present)
    def get_appearance(self, category: str, type_key: str) -> Dict[str, Any]:
        tpl = self.get_template(category, type_key) or {}
        return dict(tpl.get("appearance") or {})

    # Shortcut: random type sampling (accepts a custom RNG)
    def pick_random_type(self, category: str, rng=None) -> Optional[str]:
        import random
        rng = rng or random
        types = self.get_all_types(category)
        return rng.choice(types) if types else None
