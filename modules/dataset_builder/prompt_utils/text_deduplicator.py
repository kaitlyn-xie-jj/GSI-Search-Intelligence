#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Deduplication Tool

Only responsible for generating structured JSONL files and dictionary pool files, without dependency on HuggingFace datasets library.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class TextDeduplicator:
    """Text Deduplicator - Only maintains 'text->index' mapping"""

    def __init__(self):
        self._text_pool: Dict[str, int] = {}  # text_hash -> index
        self._texts: List[str] = []  # index -> text
        self._counter = 0

    def add_text(self, text: str) -> int:
        """Add text and return index"""
        if not text:
            text = ""

        # Ensure it's a string
        if not isinstance(text, str):
            if isinstance(text, dict) or isinstance(text, list):
                text = json.dumps(text, ensure_ascii=False)
            else:
                text = str(text)

        # Use hash as key to speed up lookup
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if text_hash in self._text_pool:
            return self._text_pool[text_hash]

        # New text
        idx = self._counter
        self._text_pool[text_hash] = idx
        self._texts.append(text)
        self._counter += 1
        return idx

    def get_all_texts(self) -> List[str]:
        """Get all text list"""
        return self._texts

    def get_stats(self) -> Dict[str, Any]:
        return {
            "unique_texts": len(self._texts),
            "total_references": len(self._text_pool),
        }


class DeduplicatedJsonlBuilder:
    """
    Deduplicated JSONL Builder

    Output:
    1. main.jsonl: Main data with index references (task-specific fields only)
    2. pool_{name}.json: Dictionary list containing actual text
    3. config.json: Global config info (planner_mode, etc.)
    """

    def __init__(
        self,
        deduplicated_fields: Dict[str, str],  # { "pool_name": "segment_key" }
        direct_fields: Optional[List[str]] = None,
        global_config: Optional[Dict[str, Any]] = None,
    ):
        self.deduplicated_fields = deduplicated_fields
        self.direct_fields = direct_fields or []
        self.global_config = global_config or {}

        # Create deduplicators
        self._deduplicators: Dict[str, TextDeduplicator] = {}
        for pool_name in deduplicated_fields.keys():
            self._deduplicators[pool_name] = TextDeduplicator()

        # Store records in memory (can be changed to write to temp file if data is extremely large)
        self._records: List[Dict[str, Any]] = []

    def add_record(
        self,
        segments: Dict[str, str],
        metadata: Dict[str, Any],
        # Accept any other additional fields
        **kwargs,
    ):
        record = {}

        # 1. Process deduplicated fields (store index)
        for pool_name, segment_key in self.deduplicated_fields.items():
            text = segments.get(segment_key, "")
            idx = self._deduplicators[pool_name].add_text(text)
            record[f"{pool_name}_idx"] = idx

        # 2. Process direct fields (store text)
        for field_name in self.direct_fields:
            record[field_name] = segments.get(field_name, "")

        # 3. Process metadata (flatten, but exclude global config fields)
        global_config_keys = {
            "planner_mode",
            "use_environment_model",
            "template_planner_mode",
            "context",  # Remove context field
        }
        for key, value in metadata.items():
            # Skip global config fields
            if key in global_config_keys:
                continue
            if isinstance(value, (dict, list)):
                record[key] = json.dumps(value, ensure_ascii=False)
            else:
                record[key] = value if value is not None else ""

        # 4. Process other parameters (exclude global config fields)
        for k, v in kwargs.items():
            if k in global_config_keys:
                continue
            record[k] = str(v) if v is not None else ""

        self._records.append(record)

    def save(self, output_dir: str, main_filename: str = "data.jsonl"):
        """Save all files to specified directory"""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save main JSONL file
        main_file = out_path / main_filename
        print(f"Saving main data: {main_file} ...")
        with open(main_file, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. Save dictionary pool files (save as JSON array for fastest loading)
        for pool_name, deduplicator in self._deduplicators.items():
            texts = deduplicator.get_all_texts()
            if not texts:
                continue

            pool_file = out_path / f"pool_{pool_name}.json"
            print(f"Saving dictionary pool: {pool_file} (entries: {len(texts)}) ...")

            # Store directly as a large JSON list ["text1", "text2", ...]
            with open(pool_file, "w", encoding="utf-8") as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)

        # 3. Save global config file
        if self.global_config:
            config_file = out_path / "config.json"
            print(f"Saving global config: {config_file} ...")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.global_config, f, ensure_ascii=False, indent=2)

    def get_stats(self):
        stats = {"total_records": len(self._records), "pools": {}}
        for name, dedup in self._deduplicators.items():
            stats["pools"][name] = dedup.get_stats()
        return stats
