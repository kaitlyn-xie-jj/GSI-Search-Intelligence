"""Collect and evaluate SearchWorld Gazebo/PX4 resource stability."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


DEFAULT_TOPICS = (
    "/gsi/rgbd/image",
    "/gsi/rgbd/camera_info",
    "/gsi/rgbd/depth_image",
    "/gsi/rgbd/points",
)
CSV_FIELDS = (
    "timestamp_utc",
    "elapsed_s",
    "container_running",
    "oom_killed",
    "container_memory_mib",
    "container_cpu_percent",
    "container_pids",
    "gazebo_rss_mib",
    "gazebo_processes",
    "px4_rss_mib",
    "px4_processes",
    "required_topics_present",
    "required_topics_active",
    "missing_topics",
    "inactive_topics",
    "topic_rates_hz_json",
    "mavros_connected",
    "mavros_armed",
    "mavros_mode",
    "mavros_x_m",
    "mavros_y_m",
    "mavros_z_m",
    "within_flight_bounds",
    "observation_count",
)


def parse_memory_mib(value: str) -> float:
    """Parse the used side of a Docker memory value into MiB."""
    used = value.split("/", 1)[0].strip()
    match = re.fullmatch(r"([0-9.]+)\s*([KMGT]?i?B)", used, re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported memory value: {value}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "b": 1.0 / (1024.0 * 1024.0),
        "kb": 1.0 / 1024.0,
        "kib": 1.0 / 1024.0,
        "mb": 1.0,
        "mib": 1.0,
        "gb": 1024.0,
        "gib": 1024.0,
        "tb": 1024.0 * 1024.0,
        "tib": 1024.0 * 1024.0,
    }
    return amount * factors[unit]


def memory_slope_mib_per_minute(samples: Sequence[Mapping[str, Any]]) -> float:
    points = [
        (float(sample["elapsed_s"]), float(sample["container_memory_mib"]))
        for sample in samples
        if sample.get("container_memory_mib") is not None
    ]
    if len(points) < 2:
        return math.nan
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0.0:
        return 0.0
    slope_per_second = sum(
        (x - mean_x) * (y - mean_y) for x, y in points
    ) / denominator
    return slope_per_second * 60.0


def evaluate_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    interval_s: float,
    max_memory_mib: float,
    max_growth_mib: float,
    max_slope_mib_per_minute: float,
    require_mavros: bool,
    require_flight: bool,
    min_observations: int,
    critical_error_count: int,
    max_critical_errors: int,
    warmup_s: float = 0.0,
) -> Dict[str, Any]:
    failures = []
    memory = [
        float(sample["container_memory_mib"])
        for sample in samples
        if sample.get("container_memory_mib") is not None
    ]
    steady_samples = [
        sample for sample in samples if float(sample.get("elapsed_s", 0.0)) >= warmup_s
    ]
    steady_memory = [
        float(sample["container_memory_mib"])
        for sample in steady_samples
        if sample.get("container_memory_mib") is not None
    ]
    elapsed = float(samples[-1]["elapsed_s"]) if samples else 0.0
    slope = memory_slope_mib_per_minute(steady_samples)
    growth = steady_memory[-1] - steady_memory[0] if len(steady_memory) >= 2 else math.nan

    if len(samples) < 2:
        failures.append("fewer than two resource samples")
    if elapsed < max(0.0, duration_s - interval_s):
        failures.append(f"test ended early at {elapsed:.1f}s")
    if any(not sample.get("container_running") for sample in samples):
        failures.append("simulator container exited")
    if any(sample.get("oom_killed") for sample in samples):
        failures.append("container was OOM-killed")
    if any(int(sample.get("gazebo_processes", 0)) < 1 for sample in samples):
        failures.append("Gazebo server was not continuously alive")
    if any(int(sample.get("px4_processes", 0)) < 1 for sample in samples):
        failures.append("PX4 was not continuously alive")
    if memory and max(memory) > max_memory_mib:
        failures.append(f"memory exceeded {max_memory_mib:.0f} MiB")
    if not math.isnan(growth) and growth > max_growth_mib:
        failures.append(f"memory growth exceeded {max_growth_mib:.0f} MiB")
    if not math.isnan(slope) and slope > max_slope_mib_per_minute:
        failures.append(
            f"memory slope exceeded {max_slope_mib_per_minute:.1f} MiB/min"
        )
    if any(not sample.get("required_topics_present") for sample in samples):
        failures.append("one or more required RGB-D topics disappeared")
    active_samples = [sample for sample in samples if sample.get("required_topics_active")]
    if not active_samples:
        failures.append("no required RGB-D topic activity was observed")
    if require_mavros:
        mavros_samples = [
            sample.get("mavros_connected")
            for sample in samples
            if sample.get("mavros_connected") is not None
        ]
        if not mavros_samples or any(value is False for value in mavros_samples):
            failures.append("MAVROS connection was missing or disconnected")
    if require_flight:
        flight_samples = [
            sample
            for sample in samples
            if sample.get("mavros_x_m") is not None
            and sample.get("mavros_y_m") is not None
            and sample.get("mavros_z_m") is not None
        ]
        if not flight_samples:
            failures.append("no MAVROS local-position samples were collected")
        elif any(sample.get("within_flight_bounds") is False for sample in flight_samples):
            failures.append("UAV left the configured SearchWorld flight bounds")
        if not any(sample.get("mavros_armed") is True for sample in samples):
            failures.append("UAV was never observed armed")
        if not any(sample.get("mavros_mode") == "OFFBOARD" for sample in samples):
            failures.append("UAV was never observed in OFFBOARD mode")
        observation_counts = [
            int(sample["observation_count"])
            for sample in samples
            if sample.get("observation_count") is not None
        ]
        if not observation_counts or max(observation_counts) < min_observations:
            failures.append(
                f"fewer than {min_observations} SearchObservation events were recorded"
            )
    if critical_error_count > max_critical_errors:
        failures.append(
            f"critical PX4/Gazebo errors {critical_error_count} > {max_critical_errors}"
        )

    return {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": len(samples),
        "elapsed_s": elapsed,
        "maximum_memory_mib": max(memory) if memory else None,
        "memory_growth_mib": None if math.isnan(growth) else growth,
        "memory_slope_mib_per_minute": None if math.isnan(slope) else slope,
        "maximum_cpu_percent": max(
            (float(sample.get("container_cpu_percent") or 0.0) for sample in samples),
            default=None,
        ),
        "critical_error_count": critical_error_count,
    }


class DockerCollector:
    def __init__(
        self,
        container: str,
        topics: Sequence[str],
        trace_path: str,
        check_mavros: bool = False,
        flight_bounds: tuple[float, float, float, float] | None = None,
    ):
        self.container = container
        self.topics = tuple(topics)
        self.trace_path = trace_path
        self.check_mavros = check_mavros
        self.flight_bounds = flight_bounds

    def _run(self, command: Sequence[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _docker_exec(self, script: str, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
        return self._run(
            ("docker", "exec", self.container, "bash", "-lc", script),
            timeout=timeout,
        )

    def collect(self, elapsed_s: float, probe_activity: bool) -> Dict[str, Any]:
        state_result = self._run(
            ("docker", "inspect", "-f", "{{json .State}}", self.container)
        )
        state = json.loads(state_result.stdout) if state_result.returncode == 0 else {}
        running = bool(state.get("Running"))

        stats: Dict[str, Any] = {}
        processes = ""
        topic_names = set()
        rates: Dict[str, float | None] = {}
        mavros = {
            "connected": None,
            "armed": None,
            "mode": "",
            "x": None,
            "y": None,
            "z": None,
        }
        observations = None
        if running:
            stats_result = self._run(
                ("docker", "stats", "--no-stream", "--format", "{{json .}}", self.container)
            )
            if stats_result.returncode == 0 and stats_result.stdout.strip():
                stats = json.loads(stats_result.stdout.splitlines()[0])
            process_result = self._docker_exec("ps -eo comm=,rss=,args=")
            processes = process_result.stdout if process_result.returncode == 0 else ""
            topics_result = self._docker_exec("gz topic -l")
            if topics_result.returncode == 0:
                topic_names = {line.strip() for line in topics_result.stdout.splitlines()}
            if probe_activity:
                with ThreadPoolExecutor(max_workers=len(self.topics)) as executor:
                    measured = executor.map(self._topic_rate, self.topics)
                    rates = dict(zip(self.topics, measured))
                if self.check_mavros:
                    mavros = self._mavros_state()
            count_result = self._docker_exec(
                f"test -f {self.trace_path!r} && "
                f"grep -c '\"event\": \"observation\"' {self.trace_path!r} || true"
            )
            if count_result.stdout.strip().isdigit():
                observations = int(count_result.stdout.strip())

        gz_rss, gz_count = self._process_totals(processes, "gazebo")
        px4_rss, px4_count = self._process_totals(processes, "px4")
        missing = [topic for topic in self.topics if topic not in topic_names]
        inactive = [topic for topic, rate in rates.items() if rate is None or rate <= 0.0]
        memory = None
        if stats.get("MemUsage"):
            memory = parse_memory_mib(str(stats["MemUsage"]))
        within_flight_bounds = None
        if self.flight_bounds is not None and mavros["x"] is not None and mavros["y"] is not None:
            min_x, min_y, max_x, max_y = self.flight_bounds
            within_flight_bounds = (
                min_x <= float(mavros["x"]) <= max_x
                and min_y <= float(mavros["y"]) <= max_y
            )

        return {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_s": round(elapsed_s, 3),
            "container_running": running,
            "oom_killed": bool(state.get("OOMKilled")),
            "container_memory_mib": memory,
            "container_cpu_percent": self._percentage(stats.get("CPUPerc")),
            "container_pids": self._integer(stats.get("PIDs")),
            "gazebo_rss_mib": round(gz_rss / 1024.0, 3),
            "gazebo_processes": gz_count,
            "px4_rss_mib": round(px4_rss / 1024.0, 3),
            "px4_processes": px4_count,
            "required_topics_present": not missing,
            "required_topics_active": bool(rates) and not inactive,
            "missing_topics": ";".join(missing),
            "inactive_topics": ";".join(inactive),
            "topic_rates_hz_json": json.dumps(rates, sort_keys=True),
            "mavros_connected": mavros["connected"],
            "mavros_armed": mavros["armed"],
            "mavros_mode": mavros["mode"],
            "mavros_x_m": mavros["x"],
            "mavros_y_m": mavros["y"],
            "mavros_z_m": mavros["z"],
            "within_flight_bounds": within_flight_bounds,
            "observation_count": observations,
        }

    def _topic_rate(self, topic: str) -> float | None:
        result = self._docker_exec(
            f"timeout 4 gz topic -t {topic!r} -f", timeout=6.0
        )
        matches = re.findall(
            r"(?:average\s+rate|hz)\s*[:=]\s*([0-9.]+)",
            result.stdout + result.stderr,
            re.IGNORECASE,
        )
        return float(matches[-1]) if matches else None

    def _mavros_state(self) -> Dict[str, Any]:
        result = self._docker_exec(
            "source /opt/ros/humble/setup.bash >/dev/null 2>&1; "
            "test ! -f /workspace/VisionFlow-PX4/thirdparty/install/setup.bash || "
            "source /workspace/VisionFlow-PX4/thirdparty/install/setup.bash >/dev/null 2>&1; "
            "test ! -f /tmp/GSI/ros2_ws/install/setup.bash || "
            "source /tmp/GSI/ros2_ws/install/setup.bash >/dev/null 2>&1; "
            "timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null || true; "
            "printf '\n---ODOMETRY---\n'; "
            "timeout 3 ros2 topic echo --once /mavros/local_position/odom "
            "2>/dev/null || true",
            timeout=8.0,
        )
        text = result.stdout
        connected = re.search(r"^connected:\s*(true|false)", text, re.MULTILINE)
        armed = re.search(r"^armed:\s*(true|false)", text, re.MULTILINE)
        mode = re.search(r"^mode:\s*['\"]?([^'\"\n]+)", text, re.MULTILINE)
        position = re.search(
            r"position:\s*\n\s*x:\s*([-+0-9.eE]+)\s*\n"
            r"\s*y:\s*([-+0-9.eE]+)\s*\n"
            r"\s*z:\s*([-+0-9.eE]+)",
            text,
        )
        return {
            "connected": connected.group(1) == "true" if connected else None,
            "armed": armed.group(1) == "true" if armed else None,
            "mode": mode.group(1).strip() if mode else "",
            "x": float(position.group(1)) if position else None,
            "y": float(position.group(2)) if position else None,
            "z": float(position.group(3)) if position else None,
        }

    @staticmethod
    def _process_totals(output: str, process_type: str) -> tuple[int, int]:
        total_rss = 0
        count = 0
        for line in output.splitlines():
            parts = line.split(maxsplit=2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            command = f"{parts[0]} {parts[2]}".lower()
            matched = (
                process_type == "gazebo"
                and any(token in command for token in ("gz sim", "gz-sim", "gzserver"))
            ) or (process_type == "px4" and re.search(r"(^|[/ ])px4([ /]|$)", command))
            if matched:
                total_rss += int(parts[1])
                count += 1
        return total_rss, count

    @staticmethod
    def _percentage(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(str(value).strip().rstrip("%"))

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def classify_log_errors(text: str) -> Dict[str, Any]:
    """Classify hard stability failures separately from diagnostic warnings."""
    critical_patterns = {
        "imu_timestamp": r"imu.*timestamp|timestamp.*imu",
        "attitude_failure": r"attitude.*(?:failure|invalid)|pitch.*(?:failure|invalid)",
        "px4_abort": r"px4.*(?:abort|segmentation fault|core dumped)",
        "out_of_memory": r"out of memory|oom-kill|oom killed",
    }
    counts = {
        name: len(re.findall(pattern, text, re.IGNORECASE))
        for name, pattern in critical_patterns.items()
    }
    warnings = {
        "mavros_timesync_reset": len(
            re.findall(r"time jump detected.*resetting time synchroniser", text, re.IGNORECASE)
        ),
    }
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "warning_counts": warnings,
        "warning_total": sum(warnings.values()),
    }


def _critical_errors(container: str, since: str, runtime_log_path: str) -> Dict[str, Any]:
    docker_result = subprocess.run(
        ("docker", "logs", "--since", since, container),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    runtime_result = subprocess.run(
        (
            "docker",
            "exec",
            container,
            "cat",
            runtime_log_path,
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    classified = classify_log_errors(
        docker_result.stdout
        + docker_result.stderr
        + runtime_result.stdout
        + runtime_result.stderr
    )
    classified["sources"] = ["docker_logs", runtime_log_path]
    return classified


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=1500.0)
    parser.add_argument("--interval-s", type=float, default=10.0)
    parser.add_argument("--activity-probe-every", type=int, default=6)
    parser.add_argument("--container", default="visionflow-px4-sitl")
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--trace-path", default="/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v1_1_trace.jsonl")
    parser.add_argument("--runtime-log-path", default="/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v1_1_runtime.log")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-memory-mib", type=float, default=8192.0)
    parser.add_argument("--max-growth-mib", type=float, default=2560.0)
    parser.add_argument("--max-slope-mib-per-minute", type=float, default=100.0)
    parser.add_argument("--warmup-s", type=float, default=120.0)
    parser.add_argument("--max-critical-errors", type=int, default=0)
    parser.add_argument("--require-mavros", action="store_true")
    parser.add_argument("--require-flight", action="store_true")
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--flight-min-x-m", type=float, default=-40.0)
    parser.add_argument("--flight-min-y-m", type=float, default=-30.0)
    parser.add_argument("--flight-max-x-m", type=float, default=40.0)
    parser.add_argument("--flight-max-y-m", type=float, default=30.0)
    options = parser.parse_args(args)
    if options.duration_s <= 0 or options.interval_s <= 0:
        parser.error("duration and interval must be positive")
    if options.activity_probe_every <= 0:
        parser.error("activity-probe-every must be positive")
    if options.min_observations < 0:
        parser.error("min-observations must be non-negative")
    if options.flight_min_x_m >= options.flight_max_x_m:
        parser.error("flight X bounds must be ordered")
    if options.flight_min_y_m >= options.flight_max_y_m:
        parser.error("flight Y bounds must be ordered")

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = options.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "resource_timeseries.csv"
    summary_path = output / "summary.json"
    started_utc = dt.datetime.now(dt.timezone.utc)
    collector = DockerCollector(
        options.container,
        options.topics or DEFAULT_TOPICS,
        options.trace_path,
        check_mavros=options.require_mavros or options.require_flight,
        flight_bounds=(
            options.flight_min_x_m,
            options.flight_min_y_m,
            options.flight_max_x_m,
            options.flight_max_y_m,
        ) if options.require_flight else None,
    )
    samples = []
    started = time.monotonic()
    next_sample = started

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        index = 0
        while True:
            now = time.monotonic()
            elapsed = now - started
            sample = collector.collect(
                elapsed,
                probe_activity=(index % options.activity_probe_every == 0),
            )
            sample["elapsed_s"] = round(time.monotonic() - started, 3)
            samples.append(sample)
            writer.writerow(sample)
            stream.flush()
            print(
                f"[{sample['elapsed_s']:7.1f}s] mem={sample['container_memory_mib']} MiB "
                f"cpu={sample['container_cpu_percent']}% gz={sample['gazebo_processes']} "
                f"px4={sample['px4_processes']} topics={sample['required_topics_present']}"
            )
            if float(sample["elapsed_s"]) >= options.duration_s:
                break
            index += 1
            next_sample = min(started + options.duration_s, next_sample + options.interval_s)
            time.sleep(max(0.0, next_sample - time.monotonic()))

    errors = _critical_errors(
        options.container,
        started_utc.isoformat(),
        options.runtime_log_path,
    )
    evaluation = evaluate_samples(
        samples,
        duration_s=options.duration_s,
        interval_s=options.interval_s,
        max_memory_mib=options.max_memory_mib,
        max_growth_mib=options.max_growth_mib,
        max_slope_mib_per_minute=options.max_slope_mib_per_minute,
        warmup_s=options.warmup_s,
        require_mavros=options.require_mavros,
        require_flight=options.require_flight,
        min_observations=options.min_observations,
        critical_error_count=errors["total"],
        max_critical_errors=options.max_critical_errors,
    )
    summary = {
        "schema_version": "1.0",
        "scenario": "gsi_search_world_v1_1",
        "started_utc": started_utc.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuration": {
            key: value
            for key, value in vars(options).items()
            if key != "output_root"
        },
        "thresholds": {
            "max_memory_mib": options.max_memory_mib,
            "max_growth_mib": options.max_growth_mib,
            "max_slope_mib_per_minute": options.max_slope_mib_per_minute,
            "warmup_s": options.warmup_s,
            "max_critical_errors": options.max_critical_errors,
        },
        "log_errors": errors,
        **evaluation,
        "timeseries_csv": csv_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['verdict']}: {summary_path}")
    raise SystemExit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
