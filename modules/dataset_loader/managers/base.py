from typing import Any, Dict, Optional

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


class BaseDataManager:
    """
    Base Data Manager

    Responsibilities:
    1. Wrap HuggingFace Dataset object
    2. Build in-memory index (Index Map) for O(1) retrieval
    """

    def __init__(self, dataset: Optional[Dataset] = None, key_column: str = "id"):
        self.ds = dataset
        self.key_column = key_column
        self._index_map: Dict[str, int] = {}

        if self.ds:
            self._build_index()

    def _build_index(self):
        """Build primary key to row index mapping"""
        print(f"   🔨 Building index ({self.key_column})...")
        try:
            keys = self.ds[self.key_column]
            self._index_map = {str(k): i for i, k in enumerate(keys)}
        except KeyError:
            print(
                f"   ⚠️ Warning: Primary key column '{self.key_column}' not found in dataset, cannot build fast index."
            )
        except Exception as e:
            print(f"   ⚠️ Index build failed: {e}")

    def get_by_id(self, key: str) -> Optional[Dict[str, Any]]:
        """Get single record by ID"""
        idx = self._index_map.get(str(key))
        if idx is not None:
            return self.ds[idx]
        return None

    def __len__(self):
        return len(self.ds) if self.ds else 0
