#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan validation server
Receives LLM-generated plan strings, validates them in the environment, and returns reward lists
Supports multi-process concurrent processing and detailed logging
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os
import sys
import time
import traceback
import uuid
import logging
import threading
from queue import Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to Python path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.plan_validator.plan_validator import validate_single_plan, validate_single_plan_async
from modules.plan_validator.replan_state_store import get_replan_state_cache_info

# Configure logging.
log_dir = Path(os.environ.get("GSI_VALIDATOR_LOG_DIR", project_root / "logs" / "plan_validation"))
log_dir.mkdir(parents=True, exist_ok=True)

# Configure root logger at WARNING level to suppress INFO logs from other modules.
logging.basicConfig(
    level=logging.WARNING,  # Record only WARNING and above, suppressing INFO logs from other modules.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create the server logger and write to both file and console.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler.
file_handler = logging.FileHandler(log_dir / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Add console handler with concise format.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Concise format that highlights key information.
console_handler.setFormatter(logging.Formatter('[SERVER] %(message)s'))
logger.addHandler(console_handler)

# Disable propagation to root logger to avoid duplicate output.
logger.propagate = False


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("%s is not an integer: %r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s is not a float: %r; using %s", name, raw, default)
        return default


def auto_tune_worker_count(cpu_count: int | None = None) -> int:
    """
    Automatically choose the validator worker count based on local CPU count.

    Lightweight startup auto-tuning to avoid the default 600 workers causing
    context switching and memory jitter. If GSI_VALIDATOR_MAX_WORKERS is set,
    the explicit value is used.
    """

    cpu_count = cpu_count or os.cpu_count() or 1
    explicit = os.environ.get("GSI_VALIDATOR_MAX_WORKERS", "").strip()
    if explicit:
        return _env_int("GSI_VALIDATOR_MAX_WORKERS", min(cpu_count, 600))
    # Full dynamic validation is CPU/memory heavy. Real vLLM-collected
    # scenario1 initial+replan benchmarks peaked around 32 concurrent requests;
    # higher defaults increased allocator/solver contention and tail latency.
    return max(1, min(cpu_count, 32))


def _env_int_list(name: str, default: List[int], *, minimum: int = 1) -> List[int]:
    raw = os.environ.get(name, "").replace(",", " ").split()
    values = []
    for item in raw:
        try:
            values.append(max(minimum, int(item)))
        except ValueError:
            logger.warning("%s contains non-integer candidate: %r", name, item)
    return values or list(default)


def _autotune_worker(plan_str: str, task_id: str = None) -> Dict[str, Any]:
    return ValidationWorker.validate_plan_worker(plan_str=plan_str, task_id=task_id)


def _run_autotune_trial(
    *,
    max_workers: int,
    batch_chunk_size: int,
    sample_size: int,
    timeout: int,
) -> Dict[str, Any]:
    sample_items = [
        {"plan": "not-json-plan", "task_id": WARMUP_PAYLOAD["task_id"]}
        for _ in range(max(1, sample_size))
    ]
    chunks = [
        sample_items[i:i + batch_chunk_size]
        for i in range(0, len(sample_items), batch_chunk_size)
    ]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(ValidationWorker.validate_plan_batch_worker, chunk)
            for chunk in chunks
        ]
        for future in as_completed(futures, timeout=max(1, timeout) * max(1, len(chunks))):
            future.result()
    elapsed = time.perf_counter() - start
    return {
        "max_workers": max_workers,
        "batch_chunk_size": batch_chunk_size,
        "sample_size": len(sample_items),
        "elapsed_sec": elapsed,
        "plans_per_sec": len(sample_items) / elapsed if elapsed > 0 else 0.0,
    }


def benchmark_worker_settings() -> Dict[str, int]:
    """
    Run a short local validation benchmark and choose the fastest worker/chunk pair.

    This uses static-invalid plans on purpose: it measures process-pool scheduling,
    JSON/static validation, and batch chunk overhead without depending on model
    quality or external services. Explicit GSI_VALIDATOR_MAX_WORKERS still bypasses
    this path through resolve_worker_settings().
    """

    cpu_count = os.cpu_count() or 1
    default_workers = sorted({max(1, cpu_count // 4), max(1, cpu_count // 2), max(1, int(cpu_count * 0.75))})
    worker_candidates = _env_int_list("GSI_VALIDATOR_AUTOTUNE_WORKERS", default_workers)
    worker_candidates = [max(1, min(cpu_count, item)) for item in worker_candidates]
    chunk_candidates = _env_int_list("GSI_VALIDATOR_AUTOTUNE_CHUNK_SIZES", [4, 8, 16])
    sample_size = _env_int("GSI_VALIDATOR_AUTOTUNE_SAMPLE_SIZE", 16)
    timeout = _env_int("GSI_VALIDATOR_TIMEOUT", 180)
    trials = []
    for worker_count in worker_candidates:
        for chunk_size in chunk_candidates:
            try:
                trials.append(
                    _run_autotune_trial(
                        max_workers=max(1, worker_count),
                        batch_chunk_size=max(1, chunk_size),
                        sample_size=sample_size,
                        timeout=timeout,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "validator autotune trial failed workers=%s chunk=%s error=%s",
                    worker_count,
                    chunk_size,
                    exc,
                )
    if not trials:
        worker_count = auto_tune_worker_count()
        chunk_size = _env_int("GSI_VALIDATOR_BATCH_CHUNK_SIZE", 8)
    else:
        best = max(trials, key=lambda item: item["plans_per_sec"])
        worker_count = int(best["max_workers"])
        chunk_size = int(best["batch_chunk_size"])
        logger.info("validator autotune selected workers=%s chunk=%s trials=%s", worker_count, chunk_size, trials)
    return {
        "max_workers": worker_count,
        "timeout": timeout,
        "queue_limit": _env_int("GSI_VALIDATOR_QUEUE_LIMIT", worker_count * 4),
        "batch_chunk_size": max(1, chunk_size),
        "autotune_sample_size": sample_size,
    }


def resolve_worker_settings(*, run_benchmark: bool = True) -> Dict[str, int]:
    explicit = os.environ.get("GSI_VALIDATOR_MAX_WORKERS", "").strip()
    if (
        run_benchmark
        and not explicit
        and os.environ.get("GSI_VALIDATOR_AUTOTUNE", "").strip().lower() in {"1", "true", "yes"}
    ):
        tuned = benchmark_worker_settings()
        tuned.pop("autotune_sample_size", None)
        return tuned
    worker_count = auto_tune_worker_count()
    return {
        "max_workers": worker_count,
        "timeout": _env_int("GSI_VALIDATOR_TIMEOUT", 180),
        "batch_timeout": _env_int("GSI_VALIDATOR_BATCH_TIMEOUT", _env_int("GSI_VALIDATOR_TIMEOUT", 180)),
        "queue_limit": _env_int("GSI_VALIDATOR_QUEUE_LIMIT", worker_count * 4),
        "batch_chunk_size": _env_int("GSI_VALIDATOR_BATCH_CHUNK_SIZE", 8),
    }


def _deduplicate_plan_requests(plans: List[Any]) -> tuple[List[Any], List[int]]:
    """Return unique plan requests and a map from original index to unique index."""

    unique: List[PlanValidationRequest] = []
    key_to_index: dict[tuple[Any, ...], int] = {}
    index_map: List[int] = []
    for item in plans:
        key = (
            item.plan,
            item.task_id,
            item.state_store,
            item.state_id,
        )
        idx = key_to_index.get(key)
        if idx is None:
            idx = len(unique)
            key_to_index[key] = idx
            unique.append(item)
        index_map.append(idx)
    return unique, index_map


@asynccontextmanager
async def lifespan(app: FastAPI):
    global process_pool, server_stats, log_thread, log_thread_running, worker_settings

    # --- Startup logic ---
    server_stats["start_time"] = datetime.now().isoformat()
    worker_settings = resolve_worker_settings()
    max_workers = worker_settings["max_workers"]
    server_stats["worker_settings"] = worker_settings

    try:
        if sys.platform != 'win32':
            # multiprocessing.set_start_method("fork", force=True)
            pass
    except RuntimeError:
        pass

    # 1. Synchronous main-process warmup using await for the async function.
    logger.info("🔥 [Main Process] Starting synchronous warmup (populating global memory)...")
    warmup_start = time.time()

    try:
        warmup_result = await validate_single_plan_async(
            WARMUP_PAYLOAD["plan"],
            WARMUP_PAYLOAD["task_id"]
        )

        duration = time.time() - warmup_start
        logger.info(f"✅ [Main Process] Warmup complete! Duration: {duration:.2f}s | Memory ready to be inherited by child processes")

    except Exception as e:
        logger.error(f"❌ Main process warmup failed: {e}")
        traceback.print_exc()

    # 2. Initialize process pool. Fork happens here and inherits the memory above.
    logger.info(f"Initializing process pool (Workers: {max_workers})...")
    process_pool = ProcessPoolExecutor(max_workers=max_workers)

    # 3. Start log thread.
    log_thread_running = True
    log_thread = threading.Thread(target=async_log_writer, daemon=False)
    log_thread.start()

    logger.info(f"Plan validation server started (Fork mode + memory sharing enabled)")
    logger.info(f"Log directory: {log_dir}")

    yield  # Server starts running and handling requests.

    # --- Shutdown logic ---
    if log_thread and log_thread_running:
        logger.info("Stopping async log thread...")
        log_thread_running = False
        log_queue.put(None)
        log_thread.join(timeout=5.0)

    if process_pool:
        process_pool.shutdown(wait=True)
        logger.info("Plan validation server shut down")
        logger.info(f"Total requests: {server_stats['total_requests']}")
        logger.info(f"Successful validations: {server_stats['successful_validations']}")


app = FastAPI(title="Plan Validation Server",
              description="Validate LLM-generated plans and return rewards",
              lifespan=lifespan)

# -----------------------------------------------------------------------------
# Warmup data, user-provided test payload.
# -----------------------------------------------------------------------------
WARMUP_PAYLOAD = {
    "plan": """```json
{
  "meta": {
    "description": "Transport the foundation base and address speaker from Hotel-1 to Intersection-1, then assemble the speaker post.",
    "shared_skill_groups": [
      ["T1.0", "T1.2", "T1.4", "T1.5", "T2.0", "T2.2", "T2.4", "T2.5", "T3.0", "T3.1"],
      ["T1.1", "T1.3", "T2.1", "T2.3"]
    ]
  },
  "task_graph": {
    "nodes": [
      {
        "task_id": "T1",
        "description": "Transport the foundation base from Hotel-1 to Intersection-1.",
        "location": "Intersection-1",
        "required_skills": [
          {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
          {"skill_name": "place<foundation_base>_on<UGV>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Intersection-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Intersection-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "place<foundation_base>_on<ground>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1}
        ],
        "produces": ["foundation_at_site"]
      },
      {
        "task_id": "T2",
        "description": "Transport the address speaker module from Hotel-1 to Intersection-1.",
        "location": "Intersection-1",
        "required_skills": [
          {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
          {"skill_name": "place<address_speaker>_on<UGV>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Intersection-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
          {"skill_name": "navigate<Intersection-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "place<address_speaker>_on<ground>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1}
        ],
        "produces": ["speaker_at_site"]
      },
      {
        "task_id": "T3",
        "description": "Install the address speaker onto the foundation base.",
        "location": "Intersection-1",
        "required_skills": [
          {"skill_name": "navigate<Intersection-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
          {"skill_name": "place<address_speaker>_on<foundation_base>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1}
        ],
        "produces": ["installation_complete"]
      }
    ],
    "edges": [
      {"from": "T1", "to": "T2", "type": "normal"},
      {"from": "T2", "to": "T3", "type": "normal"}
    ]
  }
}
```""",
    "task_id": "cybertown_scenario_1_g_0",
    "metadata": {"source": "warmup_at_startup"}
}


class PlanValidationRequest(BaseModel):
    """Plan validation request model (single plan)"""
    plan: str  # Single plan string.
    task_id: str = None
    state_store: str = None
    state_id: str = None
    metadata: Dict[str, Any] = None


class PlanValidationResponse(BaseModel):
    """Plan validation response model"""
    request_id: str
    valid: bool  # Whether the whole plan is valid.
    overall_reward: float  # Overall reward.
    token_rewards: List[float]  # Token-level reward list, indexed by plan string position.
    error_positions: List[int] = []  # Error position list.
    feedbacks: List[Dict[str, Any]] = []  # Detailed feedback information.
    timestamp: float
    metadata: Dict[str, Any] = None
    processing_time: float = None
    validation_details: Dict[str, Any] = None  # Validation details.


class PlanValidationBatchRequest(BaseModel):
    """Batch plan validation request model."""
    plans: List[PlanValidationRequest]
    metadata: Dict[str, Any] = None


class PlanValidationBatchResponse(BaseModel):
    """Batch plan validation response model."""
    request_id: str
    results: List[PlanValidationResponse]
    timestamp: float
    processing_time: float = None
    metadata: Dict[str, Any] = None


class ValidationWorker:
    """Validation worker process"""

    @staticmethod
    def validate_plan_worker(plan_str: str,
                             task_id: str = None,
                             state_store: str = None,
                             state_id: str = None) -> Dict[str, Any]:
        """
        Plan validation function in worker process

        Args:
            plan_str: Plan string
            task_id: Task ID

        Returns:
            Validation result
        """
        try:
            # Execute validation in the worker process.
            result = validate_single_plan(
                plan_str=plan_str,
                task_id=task_id,
                state_store=state_store,
                state_id=state_id,
            )
            return result
        except Exception as e:
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "reward": -1.0,
                "plan": plan_str
            }

    @staticmethod
    def validate_plan_batch_worker(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch validation function in the worker process, reducing executor scheduling calls."""

        results = []
        for item in items:
            results.append(
                ValidationWorker.validate_plan_worker(
                    plan_str=item.get("plan", ""),
                    task_id=item.get("task_id"),
                    state_store=item.get("state_store"),
                    state_id=item.get("state_id"),
                )
            )
        return results


# Global process pool.
process_pool = None
# Do not run benchmark auto-tune at module import time: uvicorn imports this
# module before startup, and forking a temporary pool during import can deadlock.
worker_settings = resolve_worker_settings(run_benchmark=False)

# Server statistics.
server_stats = {
    "total_requests": 0,
    "total_plans": 0,
    "successful_validations": 0,
    "failed_validations": 0,
    "start_time": None,
    "total_processing_time": 0.0,  # Accumulated processing time.
    "avg_processing_time": 0.0,  # Average processing time.
    "worker_settings": worker_settings,
}

# Async logging system.
log_queue = Queue(maxsize=10000)  # Log queue, caching up to 10000 entries.
log_thread = None
log_thread_running = False


def async_log_writer():
    """Background thread: asynchronously write logs to file"""
    global log_thread_running

    # Create detailed log file.
    detailed_log_file = log_dir / f"validation_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    logger.info(f"Async log thread started, log file: {detailed_log_file}")

    with open(detailed_log_file, 'a', encoding='utf-8') as f:
        while log_thread_running:
            try:
                # Get a log entry from the queue with a 1-second timeout.
                log_entry = log_queue.get(timeout=1.0)

                if log_entry is None:  # None means stop signal.
                    break

                # Write logs in JSONL format, one JSON object per line.
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written to disk.

                log_queue.task_done()

            except Exception:
                # Queue empty or timed out; continue loop.
                continue

    logger.info("Async log thread stopped")


def _response_from_worker_result(
    *,
    request: PlanValidationRequest,
    result: Dict[str, Any],
    request_id: str,
    processing_time: float,
) -> PlanValidationResponse:
    is_valid = result.get("valid", False)
    overall_reward = result.get("reward", -1.0)
    feedbacks = result.get("feedbacks", [])
    token_rewards = result.get("token_rewards", [])
    error_positions = result.get("error_positions", [])
    if not token_rewards:
        plan_length = len(request.plan) if request.plan else 1
        token_rewards = [overall_reward] * plan_length

    result_details = dict(result)
    result_details["state_store"] = request.state_store
    result_details["state_id"] = request.state_id
    result_details.setdefault("details", {})

    return PlanValidationResponse(
        request_id=request_id,
        valid=is_valid,
        overall_reward=overall_reward,
        token_rewards=token_rewards,
        error_positions=error_positions,
        feedbacks=feedbacks,
        timestamp=time.time(),
        metadata=request.metadata or {},
        processing_time=processing_time,
        validation_details=result_details,
    )


def _timeout_response(request: PlanValidationRequest, request_id: str, processing_time: float) -> PlanValidationResponse:
    error_msg = f"Validation timed out ({worker_settings['timeout']}s)"
    plan_length = len(request.plan) if request.plan else 1
    timeout_reward = _env_float("GSI_VALIDATOR_TIMEOUT_REWARD", -10.0)
    return PlanValidationResponse(
        request_id=request_id,
        valid=False,
        overall_reward=timeout_reward,
        token_rewards=[timeout_reward] * plan_length,
        error_positions=[],
        feedbacks=[{"error": error_msg}],
        timestamp=time.time(),
        metadata=request.metadata or {},
        processing_time=processing_time,
        validation_details={"error": error_msg, "state_store": request.state_store, "state_id": request.state_id},
    )


def _error_response(
    request: PlanValidationRequest,
    request_id: str,
    processing_time: float,
    error_msg: str,
) -> PlanValidationResponse:
    plan_length = len(request.plan) if request.plan else 1
    return PlanValidationResponse(
        request_id=request_id,
        valid=False,
        overall_reward=-1.0,
        token_rewards=[-1.0] * plan_length,
        error_positions=[],
        feedbacks=[{"error": error_msg}],
        timestamp=time.time(),
        metadata=request.metadata or {},
        processing_time=processing_time,
        validation_details={"error": error_msg, "state_store": request.state_store, "state_id": request.state_id},
    )


def _update_stats_for_response(response: PlanValidationResponse):
    if response.valid:
        server_stats["successful_validations"] += 1
    else:
        server_stats["failed_validations"] += 1


def _log_validation_result(request: PlanValidationRequest, response: PlanValidationResponse, request_id: str):
    task_id_str = request.task_id or "unknown"
    state_ref = f"{request.state_store}/{request.state_id}" if request.state_store and request.state_id else ""
    feedbacks = response.feedbacks or []
    reward_sources = {}
    for fb in feedbacks:
        validator_name = fb.get("validator_name") or fb.get("message", "unknown")
        fb_reward = fb.get("reward", 0.0)
        if validator_name not in reward_sources:
            reward_sources[validator_name] = {"count": 0, "total_reward": 0.0}
        reward_sources[validator_name]["count"] += 1
        reward_sources[validator_name]["total_reward"] += fb_reward
    reward_source_parts = [f"{k}={v['total_reward']:.1f}" for k, v in sorted(reward_sources.items())]
    reward_source_str = ",".join(reward_source_parts) if reward_source_parts else "none"
    logger.info(
        f"[{request_id}] task={task_id_str} "
        f"state={state_ref or 'base'} "
        f"reward={response.overall_reward:.1f} sources=[{reward_source_str}] "
        f"time={(response.processing_time or 0.0):.2f}s"
    )

    detailed_log = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id_str,
        "state_store": request.state_store,
        "state_id": request.state_id,
        "valid": response.valid,
        "overall_reward": response.overall_reward,
        "processing_time": response.processing_time,
        "timing": (response.validation_details or {}).get("details", {}).get("timing"),
        "server_stats": {
            "total_requests": server_stats["total_requests"],
            "success_rate": (
                server_stats["successful_validations"] / server_stats["total_requests"]
                if server_stats["total_requests"]
                else 0.0
            ),
        },
    }
    try:
        log_queue.put_nowait(detailed_log)
    except Exception:
        pass


@app.on_event("startup")
async def startup_event():
    """Initialize process pool, async log thread, and perform warmup on application startup"""
    global process_pool, server_stats, log_thread, log_thread_running

    # 1. Record server start time.
    server_stats["start_time"] = datetime.now().isoformat()

    worker_settings = resolve_worker_settings()
    max_workers = worker_settings["max_workers"]
    server_stats["worker_settings"] = worker_settings

    # -------------------------------------------------------------------------
    # 2. Synchronous main-process warmup before creating the process pool.
    # -------------------------------------------------------------------------
    logger.info("🔥 [Main Process] Starting synchronous warmup (populating global memory)... This may take a few seconds")
    warmup_start = time.time()

    try:
        # Call the validation function directly in the main thread.
        # This triggers _get_or_init_global_loader() in plan_validator.py
        # and loads data into the main process's _GLOBAL_TASK_CACHE.
        warmup_result = validate_single_plan(
            WARMUP_PAYLOAD["plan"],
            WARMUP_PAYLOAD["task_id"]
        )

        duration = time.time() - warmup_start
        logger.info(warmup_result)
        logger.info(f"✅ [Main Process] Warmup complete! Duration: {duration:.2f}s | Memory ready to be inherited by child processes")

    except Exception as e:
        logger.error(f"❌ Main process warmup failed (critical): {e}")
        traceback.print_exc()
        # Warmup failure usually means configuration is wrong and should be checked.

    # -------------------------------------------------------------------------
    # 3. Initialize process pool. Fork copies the main process memory loaded above.
    # -------------------------------------------------------------------------
    logger.info(f"Initializing process pool (Workers: {max_workers})...")
    process_pool = ProcessPoolExecutor(max_workers=max_workers)

    # 4. Start log thread.
    log_thread_running = True
    log_thread = threading.Thread(target=async_log_writer, daemon=False)
    log_thread.start()

    logger.info(f"Plan validation server started (Fork mode + memory sharing enabled)")
    logger.info(f"Log directory: {log_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    global process_pool, log_thread, log_thread_running

    # Stop log thread.
    if log_thread and log_thread_running:
        logger.info("Stopping async log thread...")
        log_thread_running = False
        log_queue.put(None)  # Send stop signal.
        log_thread.join(timeout=5.0)  # Wait for thread to finish.
        logger.info("Async log thread stopped")

    if process_pool:
        process_pool.shutdown(wait=True)
        logger.info("Plan validation server shut down")
        logger.info(f"Total requests: {server_stats['total_requests']}")
        logger.info(f"Total plans: {server_stats['total_plans']}")
        logger.info(f"Successful validations: {server_stats['successful_validations']}")
        logger.info(f"Failed validations: {server_stats['failed_validations']}")


@app.post("/validate", response_model=PlanValidationResponse)
async def validate_plans(request: PlanValidationRequest):
    """
    Validate a single plan and return token-level rewards (async non-blocking version)
    """
    global process_pool, server_stats

    if not process_pool:
        raise HTTPException(status_code=500, detail="Server was not initialized correctly")
    if server_stats["total_plans"] - server_stats["successful_validations"] - server_stats["failed_validations"] >= worker_settings["queue_limit"]:
        raise HTTPException(status_code=429, detail="validator queue limit exceeded")

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Update statistics.
    server_stats["total_requests"] += 1
    server_stats["total_plans"] += 1

    try:
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    process_pool,
                    ValidationWorker.validate_plan_worker,
                    request.plan,
                    request.task_id,
                    request.state_store,
                    request.state_id
                ),
                timeout=float(worker_settings["timeout"])
            )
            processing_time = time.time() - start_time
            response = _response_from_worker_result(
                request=request,
                result=result,
                request_id=request_id,
                processing_time=processing_time,
            )
        except asyncio.TimeoutError:
            response = _timeout_response(request, request_id, time.time() - start_time)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{request_id}] Worker execution error: {error_msg}")
            response = _error_response(request, request_id, time.time() - start_time, error_msg)

        # Update performance statistics.
        _update_stats_for_response(response)
        processing_time = response.processing_time or 0.0
        server_stats["total_processing_time"] += processing_time
        server_stats["avg_processing_time"] = (
                server_stats["total_processing_time"] / server_stats["total_requests"]
        )
        _log_validation_result(request, response, request_id)
        return response

    except Exception as e:
        logger.error(f"[{request_id}] CRITICAL ERROR: {str(e)}")
        # Even if this fails, return a JSON 500 response instead of plain text when possible.
        raise HTTPException(status_code=500, detail=f"Server internal error: {str(e)}")


@app.post("/validate_batch", response_model=PlanValidationBatchResponse)
async def validate_batch(request: PlanValidationBatchRequest):
    """Batch-validate plans to reduce HTTP and executor scheduling overhead."""

    global process_pool, server_stats
    if not process_pool:
        raise HTTPException(status_code=500, detail="Server was not initialized correctly")
    if not request.plans:
        return PlanValidationBatchResponse(
            request_id=str(uuid.uuid4())[:8],
            results=[],
            timestamp=time.time(),
            processing_time=0.0,
            metadata=request.metadata or {},
        )
    if len(request.plans) > worker_settings["queue_limit"]:
        raise HTTPException(status_code=429, detail="validator batch exceeds queue limit")

    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    server_stats["total_requests"] += 1
    server_stats["total_plans"] += len(request.plans)

    try:
        loop = asyncio.get_running_loop()
        chunk_size = max(1, worker_settings["batch_chunk_size"])
        unique_plans, original_to_unique = _deduplicate_plan_requests(request.plans)
        chunks = [
            unique_plans[i:i + chunk_size]
            for i in range(0, len(unique_plans), chunk_size)
        ]
        futures = []
        for chunk in chunks:
            items = [
                item.model_dump() if hasattr(item, "model_dump") else item.dict()
                for item in chunk
            ]
            futures.append(
                loop.run_in_executor(
                    process_pool,
                    ValidationWorker.validate_plan_batch_worker,
                    items,
                )
            )
        raw_chunk_results = await asyncio.wait_for(
            asyncio.gather(*futures),
            timeout=float(worker_settings.get("batch_timeout", worker_settings["timeout"])),
        )
        unique_results = [result for chunk in raw_chunk_results for result in chunk]
        if len(unique_results) != len(unique_plans):
            raise RuntimeError(
                f"validator batch result length mismatch: requested={len(unique_plans)} returned={len(unique_results)}"
            )
        processing_time = time.time() - start_time
        responses = []
        for idx, plan_request in enumerate(request.plans):
            result = unique_results[original_to_unique[idx]]
            response = _response_from_worker_result(
                request=plan_request,
                result=result,
                request_id=f"{request_id}-{idx}",
                processing_time=processing_time,
            )
            _update_stats_for_response(response)
            _log_validation_result(plan_request, response, f"{request_id}-{idx}")
            responses.append(response)

        server_stats["total_processing_time"] += processing_time
        server_stats["avg_processing_time"] = (
            server_stats["total_processing_time"] / server_stats["total_requests"]
        )
        return PlanValidationBatchResponse(
            request_id=request_id,
            results=responses,
            timestamp=time.time(),
            processing_time=processing_time,
            metadata=request.metadata or {},
        )
    except asyncio.TimeoutError:
        processing_time = time.time() - start_time
        responses = [
            _timeout_response(plan_request, f"{request_id}-{idx}", processing_time)
            for idx, plan_request in enumerate(request.plans)
        ]
        for response in responses:
            _update_stats_for_response(response)
        return PlanValidationBatchResponse(
            request_id=request_id,
            results=responses,
            timestamp=time.time(),
            processing_time=processing_time,
            metadata=request.metadata or {},
        )
    except Exception as e:
        logger.error(f"[{request_id}] batch validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch validation internal error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "stats": server_stats,
        "worker_settings": worker_settings,
        "state_cache": get_replan_state_cache_info(),
    }


@app.get("/")
async def root():
    """Root path"""
    uptime = None
    if server_stats["start_time"]:
        start = datetime.fromisoformat(server_stats["start_time"])
        uptime = (datetime.now() - start).total_seconds()

    return {
        "message": "Plan validation server is running",
        "version": "2.0",
        "uptime_seconds": uptime,
        "stats": server_stats,
        "worker_settings": worker_settings,
        "endpoints": {
            "POST /validate": "Validate plan and return reward",
            "GET /health": "Health check",
            "GET /stats": "Get server statistics",
            "GET /": "Server info"
        }
    }


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    uptime = None
    if server_stats["start_time"]:
        start = datetime.fromisoformat(server_stats["start_time"])
        uptime = (datetime.now() - start).total_seconds()

    return {
        "stats": server_stats,
        "worker_settings": worker_settings,
        "state_cache": get_replan_state_cache_info(),
        "uptime_seconds": uptime,
        "success_rate": (
            server_stats["successful_validations"] / server_stats["total_plans"]
            if server_stats["total_plans"] > 0 else 0
        )
    }


def main():
    """Main function"""
    import uvicorn

    # Set server parameters.
    host = os.environ.get("PLAN_VALIDATION_HOST", "127.0.0.1")
    port = int(os.environ.get("PLAN_VALIDATION_PORT", 8000))
    workers = int(os.environ.get("PLAN_VALIDATION_WORKERS", 1))

    print(f"Starting plan validation server: http://{host}:{port}")
    print(f"Worker count: {workers}")

    # Start server.
    uvicorn.run(
        "run.plan_validation_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,  # Disable reload in production.
        log_level="warning"  # Set uvicorn log level to warning to hide HTTP request logs.
    )


if __name__ == "__main__":
    main()
