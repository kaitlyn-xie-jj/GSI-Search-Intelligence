# modules/utils/metrics_utils.py
from typing import Any, Dict, Literal, Optional, Tuple

class MetricsManager:
    """
    A class for managing, recording, and updating experiment metrics in one place.
    It wraps metric values and update rules so callers stay concise and declarative.
    """
    def __init__(self, metric_definitions: Dict[str, Tuple[Any, Literal['set', 'increment', 'append']]]):
        """
        Initialize the metrics manager.

        Args:
            metric_definitions (Dict): Dictionary mapping metric keys to tuples.
                                      Tuple format: (initial_value, update_rule)
                                      Example: {"replans_full": (0, 'increment')}
        """
        self._metrics: Dict[str, Any] = {}
        self._rules: Dict[str, str] = {}
        
        for key, (initial_value, rule) in metric_definitions.items():
            self._metrics[key] = initial_value
            self._rules[key] = rule

    def record(self, key: str, value: Any, mode: Optional[Literal['set', 'increment', 'append']] = None) -> None:
        """
        Record a single metric.

        Args:
            key (str): Metric name.
            value (Any): Value to record.
            mode (Optional[str]): Explicit update mode. If None, use the rule defined at initialization.
        """
        effective_mode = mode or self._rules.get(key, 'set')

        try:
            if effective_mode == 'set':
                self._metrics[key] = value
            elif effective_mode == 'increment':
                self._metrics[key] = self._metrics.get(key, 0) + value
            elif effective_mode == 'append':
                self._metrics.setdefault(key, []).append(value)
        except Exception:
            pass

    def merge_from(self, source_dict: Dict[str, Any]) -> None:
        """
        Merge multiple metrics from a source dictionary, applying preset update rules.
        
        Unregistered metrics, such as algorithm-specific metrics, are automatically
        registered with the 'set' rule before merging.
        """
        for key, value in source_dict.items():
            if value is None:
                continue

            # Automatically register unregistered metrics with the 'set' rule.
            if key not in self._rules:
                self._rules[key] = 'set'
                self._metrics[key] = value
                continue

            rule = self._rules[key]
            if rule == 'append' and isinstance(value, list):
                for item in value:
                    self.record(key, item, mode='append')
            else:
                self.record(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a single metric value."""
        return self._metrics.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Get a dictionary containing all metrics."""
        return self._metrics
