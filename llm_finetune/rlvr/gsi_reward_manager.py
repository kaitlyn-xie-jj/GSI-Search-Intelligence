"""GSI RLVR reward manager backed by the local plan validation server."""

from __future__ import annotations

import functools
import logging
import os
import asyncio
from typing import Any, Mapping, Sequence

import requests
from requests.adapters import HTTPAdapter

try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except Exception:  # pragma: no cover - local unit tests may not have verl installed.
    DataProto = Any

    class RewardManagerBase:  # type: ignore[no-redef]
        def __init__(self, config: Any, tokenizer: Any, compute_score: Any):
            self.config = config
            self.tokenizer = tokenizer
            self.compute_score = compute_score


logger = logging.getLogger(__name__)
VALIDATOR_SERVER_URL = os.environ.get("GSI_VALIDATOR_URL", "http://127.0.0.1:8000/validate")
VALIDATOR_TIMEOUT = float(os.environ.get("GSI_VALIDATOR_TIMEOUT", "30"))
VALIDATOR_HTTP_TIMEOUT = float(os.environ.get("GSI_REWARD_HTTP_TIMEOUT", str(VALIDATOR_TIMEOUT + 30.0)))
VALIDATOR_TIMEOUT_REWARD = float(os.environ.get("GSI_VALIDATOR_TIMEOUT_REWARD", "-10.0"))
VALIDATOR_BATCH_CHUNK_SIZE = max(1, int(os.environ.get("GSI_VALIDATOR_BATCH_CHUNK_SIZE", "8")))


def _default_batch_url(single_url: str) -> str:
    normalized = single_url.rstrip("/")
    if normalized.endswith("/validate"):
        return normalized[: -len("/validate")] + "/validate_batch"
    return normalized + "/validate_batch"


VALIDATOR_BATCH_SERVER_URL = os.environ.get("GSI_VALIDATOR_BATCH_URL", _default_batch_url(VALIDATOR_SERVER_URL))

_SESSION = requests.Session()
_SESSION.trust_env = False
_HTTP_POOL_CONNECTIONS = max(16, int(os.environ.get("GSI_HTTP_POOL_CONNECTIONS", "128")))
_HTTP_POOL_MAXSIZE = max(16, int(os.environ.get("GSI_HTTP_POOL_MAXSIZE", "128")))
_SESSION.mount("http://", HTTPAdapter(pool_connections=_HTTP_POOL_CONNECTIONS, pool_maxsize=_HTTP_POOL_MAXSIZE))
_SESSION.mount("https://", HTTPAdapter(pool_connections=_HTTP_POOL_CONNECTIONS, pool_maxsize=_HTTP_POOL_MAXSIZE))


def _raise_validator_infra_error(message: str) -> None:
    fallback = os.environ.get("GSI_REWARD_INFRA_FALLBACK")
    if fallback is not None:
        raise RuntimeError(
            f"{message}; GSI_REWARD_INFRA_FALLBACK is no longer used for validator infrastructure errors"
        )
    raise RuntimeError(message)


def _build_plan_payload(plan_str: str, extra_info: Mapping[str, Any] | None) -> dict[str, str] | None:
    if extra_info is None:
        logger.warning("[GSIReward] missing extra_info")
        return None
    task_id = extra_info.get("task_id")
    if task_id is None:
        logger.error("[GSIReward] extra_info missing task_id; keys=%s", list(extra_info.keys()))
        return None

    payload = {"plan": plan_str, "task_id": str(task_id)}
    state_store = str(extra_info.get("state_store") or "")
    state_id = str(extra_info.get("state_id") or "")
    if state_store and state_id:
        payload["state_store"] = state_store
        payload["state_id"] = state_id
    return payload


@functools.lru_cache(maxsize=4096)
def _get_reward_from_server(plan_str: str, task_id: str, state_store: str = "", state_id: str = "") -> float:
    payload = {"plan": plan_str, "task_id": task_id}
    if state_store and state_id:
        payload["state_store"] = state_store
        payload["state_id"] = state_id
    try:
        response = _SESSION.post(VALIDATOR_SERVER_URL, json=payload, timeout=VALIDATOR_HTTP_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return float(result.get("overall_reward", 0.0))
    except requests.exceptions.ConnectionError:
        message = f"[GSIReward] validator connection failed: {VALIDATOR_SERVER_URL}"
        logger.error(message)
        _raise_validator_infra_error(message)
    except requests.exceptions.Timeout:
        message = f"[GSIReward] validator timeout: task_id={task_id} state={state_store}/{state_id}"
        logger.error(message)
        return VALIDATOR_TIMEOUT_REWARD
    except Exception as exc:
        message = f"[GSIReward] validator error: task_id={task_id} state={state_store}/{state_id} error={exc}"
        logger.error(message)
        _raise_validator_infra_error(message)


def cybertown_score_fn(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
) -> float:
    payload = _build_plan_payload(str(solution_str or ""), extra_info)
    if payload is None:
        return 0.0
    return _get_reward_from_server(
        payload["plan"],
        payload["task_id"],
        payload.get("state_store", ""),
        payload.get("state_id", ""),
    )


def _fallback_batch_scores(solutions: Sequence[str], extras: Sequence[Mapping[str, Any] | None]) -> list[float]:
    return [
        cybertown_score_fn(solution_str=solution, extra_info=extra)
        for solution, extra in zip(solutions, extras, strict=True)
    ]


def cybertown_score_fn_batched(
    data_sources: Sequence[str] | None = None,
    solution_strs: Sequence[str] | None = None,
    ground_truths: Sequence[Any] | None = None,
    extra_infos: Sequence[Mapping[str, Any] | None] | None = None,
    **_: Any,
) -> list[float]:
    solutions = [str(item or "") for item in solution_strs] if solution_strs is not None else []
    extras = list(extra_infos) if extra_infos is not None else []
    if len(extras) < len(solutions):
        extras.extend([None] * (len(solutions) - len(extras)))
    elif len(extras) > len(solutions):
        extras = extras[: len(solutions)]

    unique_plans: list[dict[str, str]] = []
    unique_keys: dict[tuple[tuple[str, str], ...], int] = {}
    result_positions_by_unique: list[list[int]] = []
    scores = [0.0] * len(solutions)
    for idx, (solution, extra) in enumerate(zip(solutions, extras, strict=True)):
        payload = _build_plan_payload(solution, extra)
        if payload is None:
            continue
        key = tuple(sorted(payload.items()))
        unique_idx = unique_keys.get(key)
        if unique_idx is None:
            unique_idx = len(unique_plans)
            unique_keys[key] = unique_idx
            unique_plans.append(payload)
            result_positions_by_unique.append([])
        result_positions_by_unique[unique_idx].append(idx)

    if not unique_plans:
        return scores

    try:
        chunk_count = (len(unique_plans) + VALIDATOR_BATCH_CHUNK_SIZE - 1) // VALIDATOR_BATCH_CHUNK_SIZE
        response = _SESSION.post(
            VALIDATOR_BATCH_SERVER_URL,
            json={"plans": unique_plans},
            timeout=VALIDATOR_HTTP_TIMEOUT * max(1, chunk_count),
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if len(results) != len(unique_plans):
            _raise_validator_infra_error(
                f"[GSIReward] validator batch length mismatch: requested={len(unique_plans)} returned={len(results)}"
            )

        for positions, result in zip(result_positions_by_unique, results, strict=True):
            score = float(result.get("overall_reward", 0.0))
            for idx in positions:
                scores[idx] = score
        return scores
    except requests.exceptions.ConnectionError:
        message = f"[GSIReward] validator batch connection failed: {VALIDATOR_BATCH_SERVER_URL}"
        logger.error(message)
        _raise_validator_infra_error(message)
    except requests.exceptions.Timeout:
        message = f"[GSIReward] validator batch timeout: batch_size={len(unique_plans)}"
        logger.error(message)
        return [VALIDATOR_TIMEOUT_REWARD] * len(solutions)
    except Exception as exc:
        message = f"[GSIReward] validator batch error: batch_size={len(unique_plans)} error={exc}"
        logger.error(message)
        _raise_validator_infra_error(message)


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: Mapping[str, Any] | None = None,
) -> float:
    return cybertown_score_fn(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


class GsiBatchRewardManager(RewardManagerBase):
    """VeRL reward manager that calls the GSI validator batch endpoint."""

    def __init__(
        self,
        config: Any,
        tokenizer: Any,
        compute_score: Any,
        reward_router_address: str | None = None,
        reward_model_tokenizer: Any = None,
        **_: Any,
    ):
        super().__init__(config, tokenizer, compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        if not hasattr(self, "loop"):
            self.loop = asyncio.get_event_loop()

    @staticmethod
    def _as_list(value: Any, length: int, default: Any = None) -> list[Any]:
        if value is None:
            return [default for _ in range(length)]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if len(value) < length:
                return value + [default for _ in range(length - len(value))]
            return value[:length]
        return [value for _ in range(length)]

    async def _decode_responses(self, data: DataProto) -> list[str]:
        batch = getattr(data, "batch", {})
        responses = batch.get("responses")
        if responses is None:
            non_tensor_batch = getattr(data, "non_tensor_batch", {})
            responses = non_tensor_batch.get("responses", [])

        try:
            response_count = len(responses)
        except TypeError:
            response_count = len(data)

        decode_items = responses
        attention_mask = batch.get("attention_mask")
        if responses is not None and hasattr(responses, "shape") and len(responses.shape) >= 2:
            response_length = responses.shape[-1]
            if attention_mask is not None:
                valid_lengths = attention_mask[:, -response_length:].sum(dim=-1).tolist()
            else:
                valid_lengths = [response_length for _ in range(response_count)]
            decode_items = [
                responses[idx][: int(valid_lengths[idx])]
                for idx in range(response_count)
            ]

        if hasattr(self.tokenizer, "batch_decode"):
            return await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.batch_decode(decode_items, skip_special_tokens=True),
            )
        return [str(item) for item in decode_items]

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        outputs = await self.run_batch(data)
        if not outputs:
            return {"reward_score": 0.0, "reward_extra_info": {"acc": 0.0}}
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        non_tensor_batch = getattr(data, "non_tensor_batch", {})
        solution_strs = await self._decode_responses(data)
        batch_size = len(solution_strs)

        raw_extra_infos = self._as_list(non_tensor_batch.get("extra_info"), batch_size, {})
        tool_extra_fields = self._as_list(non_tensor_batch.get("tool_extra_fields"), batch_size, None)
        num_turns = self._as_list(non_tensor_batch.get("__num_turns__"), batch_size, None)
        rollout_reward_scores = self._as_list(non_tensor_batch.get("reward_scores"), batch_size, {})
        data_sources = self._as_list(non_tensor_batch.get("data_source"), batch_size, None)
        reward_models = self._as_list(non_tensor_batch.get("reward_model"), batch_size, {})

        extra_infos: list[Mapping[str, Any] | None] = []
        ground_truths: list[Any] = []
        for idx in range(batch_size):
            raw_extra = raw_extra_infos[idx]
            extra = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}

            tool_extra = tool_extra_fields[idx]
            if isinstance(tool_extra, Mapping):
                extra.update(tool_extra)

            if num_turns[idx] is not None:
                extra["num_turns"] = num_turns[idx]
            if rollout_reward_scores[idx] is not None:
                extra["rollout_reward_scores"] = rollout_reward_scores[idx]

            extra_infos.append(extra)

            reward_model = reward_models[idx]
            if isinstance(reward_model, Mapping):
                ground_truths.append(reward_model.get("ground_truth"))
            else:
                ground_truths.append(None)

        scores = await self.loop.run_in_executor(
            None,
            lambda: cybertown_score_fn_batched(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
            ),
        )

        outputs: list[dict[str, Any]] = []
        for score in scores:
            reward = float(score)
            outputs.append({"reward_score": reward, "reward_extra_info": {"acc": reward}})
        return outputs

    def __call__(self, data: DataProto, return_dict: bool = False) -> Any:
        batch = getattr(data, "batch", {})
        non_tensor_batch = getattr(data, "non_tensor_batch", {})
        responses = batch.get("responses")
        if responses is None:
            responses = non_tensor_batch.get("responses", [])

        if hasattr(self.tokenizer, "batch_decode"):
            solution_strs = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        else:
            solution_strs = [str(item) for item in responses]

        extra_infos = non_tensor_batch.get("extra_info", [{} for _ in solution_strs])
        scores = cybertown_score_fn_batched(solution_strs=solution_strs, extra_infos=extra_infos)
        if return_dict:
            return {"reward_tensor": scores}
        return scores
