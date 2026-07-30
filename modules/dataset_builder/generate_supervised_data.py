import os
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_dataset_from_logs(root_dir, output_file, target_batches=None):
    """
    Args:
        root_dir: Root directory path
        output_file: Output file path
        target_batches: (List[str]) Filter keywords
    """
    dataset = []
    processed_count = 0
    skipped_count = 0

    print(f"Starting directory scan: {root_dir}")
    if target_batches:
        print(f"Filter keywords set: {target_batches}")

    # Walk through all files under root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # --- Filter logic ---
        if target_batches:
            is_target = any(batch_name in dirpath for batch_name in target_batches)
            if not is_target:
                continue

        target_file = 'temp_vars.jsonl'

        if target_file in filenames:
            file_path = os.path.join(dirpath, target_file)

            # Data container initialization
            current_data = {
                "task_type": "unknown",
                "scenario_id": None,
                "goal_id": None,
                "prompt": None,
                "response": None,
                "label": None
            }

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            name = record.get("name")
                            value = record.get("value")

                            # 1. Get metadata (type_name, scenario_id, goal_id)
                            if name == "run_input_default" and isinstance(value, dict):
                                current_data["task_type"] = value.get("type_name", "unknown")
                                current_data["scenario_id"] = value.get("scenario_id")
                                current_data["goal_id"] = value.get("goal_id")

                            # 2. Get Prompt
                            if name == "prompt" and current_data["prompt"] is None:
                                current_data["prompt"] = value

                            # 3. Get Response
                            if name == "response" and current_data["response"] is None:
                                current_data["response"] = value

                            # 4. Get execution result (Label)
                            if name == "execution_result" and isinstance(value, dict):
                                status = value.get("status")
                                # Unify bool or string handling to ensure we get the execution status
                                current_data["label"] = status

                        except json.JSONDecodeError:
                            continue

                # --- Completeness check and error reporting ---
                missing_fields = []
                if not current_data["prompt"]: missing_fields.append("prompt")
                if not current_data["response"]: missing_fields.append("response")
                if current_data["label"] is None: missing_fields.append("label")
                # If you need to ensure IDs must exist, add checks here too
                if not current_data["scenario_id"]: missing_fields.append("scenario_id")

                if not missing_fields:
                    # Build task_id string (e.g.: cybertown_scenario_1_goal_1)
                    task_type = current_data["task_type"]
                    s_id = current_data["scenario_id"]
                    g_id = current_data["goal_id"]

                    # Compose task_id
                    if s_id and g_id:
                        full_task_id = f"{task_type}_{s_id}_{g_id}"
                    else:
                        full_task_id = f"{task_type}_unknown_unknown"

                    # Build output format
                    entry = {
                        "task_type": task_type,
                        "task_id": full_task_id,  # Added task_id
                        "prompt": [{"role": "user", "content": current_data["prompt"]}],
                        "completion": [{"role": "assistant", "content": current_data["response"]}],
                        "label": current_data["label"]
                    }
                    dataset.append(entry)
                    processed_count += 1
                else:
                    # Print specific skip reason
                    skipped_count += 1
                    # Simplify path display, show only last two levels
                    short_path = os.path.join(os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath))
                    print(f"[Skipped] {short_path} | Reason: Missing {missing_fields}")

            except Exception as e:
                print(f"[Error] Failed to read file: {file_path}, Error: {e}")

    # Write results
    print("-" * 60)
    print(f"Scan complete.")
    print(f"Successfully extracted: {processed_count} records")
    print(f"Skipped/Invalid: {skipped_count} records")

    if processed_count > 0:
        print(f"Writing to: {output_file} ...")
        # Fixed previous mkdir error
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for item in dataset:
                out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Write complete!")
    else:
        print("No valid data extracted, no file written.")


if __name__ == "__main__":
    INPUT_ROOT_DIRECTORY = r"/home/yons/GSI/results/batch_runs"
    OUTPUT_FILENAME = os.path.join(
        os.path.dirname(__file__),
        "../../dataset/semantic/prompts/supervised_prompt.jsonl",
    )

    TARGET_BATCH_FOLDERS = [
        # "batch_2025-12-11_13-53-29",
        "batch_2025-12-11_21-03-54"
    ]

    extract_dataset_from_logs(INPUT_ROOT_DIRECTORY, OUTPUT_FILENAME, TARGET_BATCH_FOLDERS)