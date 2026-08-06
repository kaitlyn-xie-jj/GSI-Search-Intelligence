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
from .unified_benchmark import (
    BASELINE_LABELS,
    UNIFIED_POLICIES,
    run_unified_benchmark,
    search_skill_scenario_matrix,
    unified_benchmark_config,
    write_unified_evaluation_report,
)
from .hybrid_experiment import (
    HYBRID_EXPERIMENT_POLICIES,
    run_hybrid_search_experiment,
    write_hybrid_search_experiment,
)
from .success_first_experiment import (
    HIGH_RES_CAMERA_PROFILE,
    SUCCESS_FIRST_VARIANTS,
    SuccessFirstVariant,
    run_success_first_experiment,
    write_success_first_experiment,
)
from .failure_analysis import (
    FAILURE_TAXONOMY,
    analyze_detection_failures,
    write_detection_failure_analysis,
)
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
    "HYBRID_EXPERIMENT_POLICIES",
    "HIGH_RES_CAMERA_PROFILE",
    "FAILURE_TAXONOMY",
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
    "SUCCESS_FIRST_VARIANTS",
    "SuccessFirstVariant",
    "SUPPORTED_POLICIES",
    "BASELINE_LABELS",
    "UNIFIED_POLICIES",
    "UtilityWeights",
    "WeightCandidateResult",
    "aggregate_policy_results",
    "analyze_detection_failures",
    "compare_paired_policy_results",
    "default_benchmark_scenarios",
    "default_offline_splits",
    "default_stress_profiles",
    "realism_stress_profiles",
    "estimate",
    "focused_grid_belief",
    "generate_weight_candidates",
    "run_stress_benchmark",
    "run_unified_benchmark",
    "run_hybrid_search_experiment",
    "run_success_first_experiment",
    "search_skill_scenario_matrix",
    "stress_benchmark_scenarios",
    "verification_stress_profiles",
    "write_benchmark_report",
    "write_offline_optimization_result",
    "write_stress_benchmark_results",
    "unified_benchmark_config",
    "write_unified_evaluation_report",
    "write_hybrid_search_experiment",
    "write_success_first_experiment",
    "write_detection_failure_analysis",
]
