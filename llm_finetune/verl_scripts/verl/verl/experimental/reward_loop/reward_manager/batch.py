# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """
    业务语义:
    - 这是 experimental reward loop 使用的批量 reward manager，用于把一个 rollout chunk
      作为整体交给 custom reward function。

    上游来源:
    - `RewardLoopWorker.compute_score_batch()` 传入同一个 Ray reward worker 负责的一批样本。
    - `custom_reward_function.name` 应指向支持 `data_sources/solution_strs/ground_truths/extra_infos`
      批量参数的函数。

    输出去向:
    - 返回值会被 `RewardLoopManager.compute_rm_score()` 转成 `rm_scores` 和 reward extra info。

    关键规则:
    - 单条 `run_single()` 保持兼容；真正的并发入口是 `run_batch()`。
    - 这里只做 VeRL batch 字段解码和 reward payload 对齐，不解释机器人任务语义。

    当前限制:
    - 这个 manager 面向 rule/replay reward，不支持 reward model router 的 batch 推理。
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """
        兼容 experimental reward loop 的单条 manager contract。

        输入仍是长度为 1 的 `DataProto`；输出保持 `reward_score/reward_extra_info` 结构。
        """

        assert len(data) == 1, "BatchRewardManager.run_single only supports one data item"
        return (await self.run_batch(data))[0]

    async def run_batch(self, data: DataProto) -> list[dict[str, Any]]:
        """
        批量计算一个 rollout chunk 的 replay reward。

        输入是同一个 reward worker 拿到的 `DataProto` chunk；输出顺序必须与输入样本顺序完全一致，
        因为下游按索引把 reward 写回 `rm_scores`。
        """

        payload = await self.loop.run_in_executor(None, lambda: self._collect_batch_payload(data))
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            results = await self.compute_score(**payload, **extra_reward_kwargs)
        else:
            results = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(**payload, **extra_reward_kwargs),
            )
        return self._normalize_results(results, expected_count=len(data))

    def _collect_batch_payload(self, data: DataProto) -> dict[str, list[Any]]:
        """
        从 VeRL `DataProto` chunk 中抽出批量 reward 函数需要的四组字段。

        这里复用 naive manager 的 response 解码规则：只解码 attention mask 标记为有效的 response token。
        """

        data_sources: list[Any] = []
        solution_strs: list[str] = []
        ground_truths: list[Any] = []
        extra_infos: list[dict[str, Any]] = []

        for index in range(len(data)):
            data_item = data[index]
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            data_source = data_item.non_tensor_batch["data_source"]
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
            tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
            if isinstance(tool_extra_fields, Mapping):
                extra_info.update(tool_extra_fields)

            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})

            data_sources.append(data_source)
            solution_strs.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

        return {
            "data_sources": data_sources,
            "solution_strs": solution_strs,
            "ground_truths": ground_truths,
            "extra_infos": extra_infos,
        }

    def _normalize_results(self, results: Any, *, expected_count: int) -> list[dict[str, Any]]:
        """
        把 custom batch reward 函数的返回值归一成 reward loop contract。

        每条结果可以是 float，也可以是包含 `score` 的 dict；dict 中的其他字段会继续作为
        reward extra info 进入训练日志。
        """

        if not isinstance(results, list):
            raise TypeError("Batch reward function must return a list")
        if len(results) != expected_count:
            raise ValueError(f"Batch reward result count mismatch: expected {expected_count}, got {len(results)}")

        outputs: list[dict[str, Any]] = []
        for result in results:
            reward_extra_info: dict[str, Any] = {}
            if isinstance(result, Mapping):
                reward = float(result["score"])
                reward_extra_info.update(dict(result))
            else:
                reward = float(result)
                reward_extra_info["acc"] = reward
            outputs.append({"reward_score": reward, "reward_extra_info": reward_extra_info})
        return outputs
