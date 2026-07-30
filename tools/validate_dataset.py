#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import asyncio
import sys
import argparse
from collections import Counter
from pathlib import Path
import aiohttp

# ================= 配置区域 =================
# 验证服务器地址
SERVER_URL = "http://127.0.0.1:8000/validate"
# 并发数量 (根据你的服务器性能调整)
CONCURRENCY = 32


# ===========================================

def clean_plan_string(content):
    """
    Clean LLM output, remove Markdown code block markers, extract pure JSON.
    """
    if not isinstance(content, str):
        return str(content)

    # 匹配 ```json ... ``` 或 ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip()
    return content.strip()


async def validate_plan_remote(session, plan_str, task_id):
    """
    Send a plan to the server for validation
    """
    clean_plan = clean_plan_string(plan_str)

    payload = {
        "plan": clean_plan,
        "task_id": task_id,
        "metadata": {
            "source": "dataset_validator"
        }
    }

    try:
        async with session.post(SERVER_URL, json=payload) as response:
            if response.status != 200:
                # print(f"Server Error ({response.status}) for {task_id}")
                return None
            return await response.json()
    except Exception as e:
        print(f"Connection Error: {e}")
        return None


async def process_line(sem, session, line, index, mismatch_counter):
    """
    Process a single line of data
    """
    async with sem:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Line {index}: JSON Decode Error")
            return None

        # 1. 提取 Plan
        plan_str = ""
        # 情况 A: 标准 OpenAI 格式 {"completion": [{"content": "..."}]}
        if "completion" in data and isinstance(data["completion"], list) and len(data["completion"]) > 0:
            plan_str = data["completion"][0].get("content", "")
        # 情况 B: 扁平格式 {"response": "..."}
        elif "response" in data:
            plan_str = data["response"]
        # 情况 C: 你的数据生成脚本可能使用的格式
        elif "completion" in data and isinstance(data["completion"], str):
            plan_str = data["completion"]

        if not plan_str:
            # print(f"Line {index}: No plan found")
            return None

        # 2. 提取 Ground Truth (Label)
        # 你的生成脚本中，batch runs 的 success 变成了 label
        ground_truth = data.get("label", False)

        # 3. 构建 Task ID (用于服务器加载地图)
        # 我们需要从 meta 中提取 scenario_id 和 goal_id
        task_id = data.get("task_id", {})


        # 4. 调用验证
        server_response = await validate_plan_remote(session, plan_str, task_id)

        if not server_response:
            return None

        # 5. 解析结果
        val_valid = server_response.get("valid", False)
        validation_details = server_response.get("validation_details", {}) or {}

        # 获取 Validator 认为的目标完成情况
        val_goal_completed = False
        if validation_details.get("details"):
            val_goal_completed = validation_details["details"].get("goal_completed", False)
        elif "goal_completed" in validation_details:
            val_goal_completed = validation_details.get("goal_completed", False)

        # 6. 核心对比: 数据集里的 Label (实际运行结果) vs 验证器预测
        match = (ground_truth == val_goal_completed)

        if not match:
            mismatch_counter[str(goal_type)] += 1

        # 格式化输出 ID，太长则截断
        display_id = f"Line_{index}"
        if scenario_id and goal_id:
            full_id = f"{scenario_id}_{goal_id}"
            if len(full_id) < 35:
                display_id = full_id

        print(
            f"{display_id:<40} | {str(ground_truth):<6} | {str(val_valid):<6} | {str(val_goal_completed):<6} | {str(match):<5}")

        return {
            "index": index,
            "ground_truth": ground_truth,
            "validator_goal_completed": val_goal_completed,
            "match": match
        }


async def main():
    parser = argparse.ArgumentParser(description="Dataset (.jsonl) Plan Validator")
    parser.add_argument("file", type=str, help="Path to the .jsonl dataset file")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return

    print(f"Target Dataset: {args.file}")
    print(f"Connecting to Server: {SERVER_URL}")
    print("-" * 88)
    # 表头含义:
    # Label = 数据集中的 ground truth (实际执行是否成功)
    # Valid = 验证器检查语法是否合法
    # V.Goal = 验证器预测目标是否会完成
    # Match = Label 是否等于 V.Goal
    print(f"{'ID / Line':<40} | {'Label':<6} | {'Valid':<6} | {'V.Goal':<6} | {'Match':<5}")
    print("-" * 88)

    mismatch_counter = Counter()
    sem = asyncio.Semaphore(CONCURRENCY)

    # 读取文件
    with open(args.file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_line(sem, session, line, idx, mismatch_counter)
            for idx, line in enumerate(lines)
        ]

        results_raw = await asyncio.gather(*tasks)

    results = [r for r in results_raw if r is not None]

    # 统计
    processed_count = len(results)
    matches = sum(1 for r in results if r['match'])
    mismatches = processed_count - matches

    print("-" * 88)
    print(f"Processed: {processed_count}/{total_lines}")
    print(f"Matches:   {matches}")
    print(f"Mismatch:  {mismatches}")

    if mismatches > 0:
        print("\n" + "=" * 40)
        print("Mismatch Distribution by Goal Type:")
        print("=" * 40)
        for g_type, count in mismatch_counter.most_common():
            percentage = (count / mismatches) * 100
            print(f"{g_type:<20} | {count:<5} | {percentage:.1f}%")
        print("=" * 40)
    else:
        print("\nPerfect Match! Validator logic aligns perfectly with dataset labels.")


if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")