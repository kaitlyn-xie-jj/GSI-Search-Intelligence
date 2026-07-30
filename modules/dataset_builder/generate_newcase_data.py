import json
import sys
import os
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_loader import DatasetLoader
from modules.utils.process.dataset_visulizer import get_and_print_distribution


def parse_cyber_town_log(file_path):
    """
    Read and parse logs, generate statistical report, and extract world_state and prompt
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Reading file: {file_path}...\n")

    # Configure output paths
    # 1. Base path for saving scenes
    scenarios_base_dir = os.path.join(os.path.dirname(__file__), "../../dataset/semantic/scenarios/cybertown")
    # 2. Save path for new prompts
    prompt_output_file = os.path.join(os.path.dirname(__file__), "../../dataset/semantic/prompts/newcase_prompt/newcase_prompt.jsonl")

    # Ensure prompt output directory exists
    os.makedirs(os.path.dirname(prompt_output_file), exist_ok=True)

    # Initialize statistics containers
    task_type_counter = Counter()
    event_type_counter = Counter()
    all_task_ids = []

    valid_records = 0

    # Initialize scenario counter (n > 1, starting from 2)
    scenario_counter = 2

    # Open input file for reading, and output file for writing prompts
    with open(file_path, 'r', encoding='utf-8') as f_in, \
            open(prompt_output_file, 'w', encoding='utf-8') as f_prompt_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line: continue

            try:
                entry = json.loads(line)
                case_info = entry.get('case', {})

                # Filter: only count records where newcase_total_orig is not 0
                if not case_info.get('newcase_total_orig', 0):
                    continue

                valid_records += 1

                # Extract basic information
                timestamp = entry.get('timestamp', 'Unknown Time')
                scenario_id = f"{case_info.get('type_name', 'N/A')}_{case_info.get('scenario_id', 'N/A')}"

                # Get original ID for statistics
                orig_goal_id = case_info.get('goal_id', 'UnknownID')
                goal_type = case_info.get('goal_type', 'Unknown Type')
                #
                # # [Requirement 1] Save World State to specified directory
                # world_state = entry.get('world_state')
                # if world_state:
                #     # Construct path: .../scenario_{n}/scene_graph.json
                #     current_scenario_dir = os.path.join(scenarios_base_dir, f"scenario_{scenario_counter}")
                #     os.makedirs(current_scenario_dir, exist_ok=True)
                #
                #     scene_graph_path = os.path.join(current_scenario_dir, "scene_graph.json")
                #     with open(scene_graph_path, 'w', encoding='utf-8') as f_sg:
                #         json.dump(world_state, f_sg, ensure_ascii=False, indent=2)
                # else:
                #     print(f"Warning: Line {line_num} missing world_state, skipping file save")

                # [Requirement 2] Construct new ID and save Prompt
                # Construct new Scenario ID and Task ID
                new_scenario_id = f"scenario_{scenario_counter}"
                new_task_id = f"{new_scenario_id}_{orig_goal_id}"
                old_task_id = f"{scenario_id}_{orig_goal_id}"

                # Get Prompt (prefer explicit prompt, fall back to instruction if not available)
                prompt_content = entry.get('prompt', entry.get('meta', {}).get('instruction', ''))

                # Write to jsonl
                new_prompt_entry = {
                    "task_id": new_task_id,
                    "prompt": prompt_content,
                    "goal_type": goal_type,
                }
                f_prompt_out.write(json.dumps(new_prompt_entry, ensure_ascii=False) + "\n")

                # Original statistics logic (unchanged)
                task_type_counter[goal_type] += 1
                all_task_ids.append(old_task_id)  # This list can record new or old IDs depending on what you need to track

                event = entry.get('event', {})
                event_type = event.get('reason', 'Unknown Reason')
                event_msg = event.get('details', {}).get('message', 'No details')
                event_type_counter[event_type] += 1

                meta_goal = entry.get('meta', {}).get('goal', {})
                goal_desc = meta_goal.get('description', 'No goal description')

                # Print progress (optional: add new save path hint)
                print(f"=== Record #{line_num} -> Saved to {new_scenario_id} ===")

                # Increment counter
                scenario_counter += 1

            except json.JSONDecodeError:
                print(f"❌ Line {line_num} JSON format error")
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {str(e)}")

    # Output statistics report
    print("\n" + "=" * 60)
    print("📊 Processing complete & Statistical report")
    print("=" * 60)
    print(f"✅ Total processed records: {valid_records}")
    print(f"📂 Scene files saved to: {scenarios_base_dir}/scenario_2 ... scenario_{scenario_counter - 1}")
    print(f"📝 Prompt file generated at: {prompt_output_file}")

    print("\n📌 1. Task Type Distribution:")
    for t_type, count in task_type_counter.most_common():
        percentage = (count / valid_records) * 100 if valid_records > 0 else 0
        print(f"{t_type:<25} | {count:<8} | {percentage:.1f}%")

    print("\n📌 2. Trigger Event Distribution:")
    for e_type, count in event_type_counter.most_common():
        percentage = (count / valid_records) * 100 if valid_records > 0 else 0
        e_name = (e_type[:37] + '..') if len(e_type) > 37 else e_type
        print(f"{e_name:<40} | {count:<8} | {percentage:.1f}%")

    print("\n📌 3. Original Dataset Distribution:")
    try:
        # Note: if all_task_ids contains newly generated IDs, the loader may not find the corresponding original data
        # If you need the original distribution, the loader logic may need adjustment, or ignore errors here
        loader = DatasetLoader(repo_id="WindyLab/GSI")
        get_and_print_distribution(loader, all_task_ids)
    except Exception as e:
        print(f"⚠️ Unable to call external loader for distribution statistics: {e}")

    print("=" * 60)


if __name__ == "__main__":
    file_name = os.path.join(os.path.dirname(__file__), "../../results/replan_batch_runs/", "batch_2026-02-25_10-46-00", "replan_dataset.jsonl")
    parse_cyber_town_log(file_name)