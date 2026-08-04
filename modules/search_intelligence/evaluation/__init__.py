"""Reproducible evaluation harness for search policies."""

from .contracts import (
    MetricEstimate,
    PolicyAggregate,
    SearchBenchmarkConfig,
    SearchBenchmarkReport,
    SearchBenchmarkScenario,
    SearchEpisodeResult,
    SUPPORTED_POLICIES,
)
from .defaults import default_benchmark_scenarios, focused_grid_belief
from .offline_optimization import (
    DEFAULT_UTILITY_WEIGHTS,
    OfflineOptimizationConfig,
    OfflineOptimizationResult,
    OfflineUtilityOptimizer,
    OptimizationScore,
    PairedValidationComparison,
    UtilityWeights,
    WeightCandidateResult,
    default_offline_splits,
    generate_weight_candidates,
    write_offline_optimization_result,
)
from .reporting import aggregate_policy_results, estimate, write_benchmark_report
from .paired_comparison import PairedMetricComparison, compare_paired_policy_results
from .runner import SearchBenchmarkRunner, SearchEpisodeRunner
from .stress import (
    SearchStressProfile,
    SearchStressRun,
    default_stress_profiles,
    realism_stress_profiles,
    run_stress_benchmark,
    stress_benchmark_scenarios,
    verification_stress_profiles,
    write_stress_benchmark_results,
)

__all__ = [
    "MetricEstimate",
    "DEFAULT_UTILITY_WEIGHTS",
    "OfflineOptimizationConfig",
    "OfflineOptimizationResult",
    "OfflineUtilityOptimizer",
    "OptimizationScore",
    "PairedValidationComparison",
    "PairedMetricComparison",
    "PolicyAggregate",
    "SearchBenchmarkConfig",
    "SearchBenchmarkReport",
    "SearchBenchmarkRunner",
    "SearchBenchmarkScenario",
    "SearchEpisodeResult",
    "SearchEpisodeRunner",
    "SearchStressProfile",
    "SearchStressRun",
    "SUPPORTED_POLICIES",
    "UtilityWeights",
    "WeightCandidateResult",
    "aggregate_policy_results",
    "compare_paired_policy_results",
    "default_benchmark_scenarios",
    "default_offline_splits",
    "default_stress_profiles",
    "realism_stress_profiles",
    "estimate",
    "focused_grid_belief",
    "generate_weight_candidates",
    "run_stress_benchmark",
    "stress_benchmark_scenarios",
    "verification_stress_profiles",
    "write_benchmark_report",
    "write_offline_optimization_result",
    "write_stress_benchmark_results",
]
