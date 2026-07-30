import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_level_dir = os.path.join(current_dir)
sys.path.append(upper_level_dir)
from typing import Optional, Dict, Tuple, Any
from datetime import datetime
from .algorithm.TeamPlannerBasic import TeamPlanner 

def tango(run_dir: Optional[str] = None, config_dicts: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    if run_dir is None:
        # If no input directory is provided, create a timestamped results directory next to this script.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        planner_yaml_dir = os.path.join(current_dir, 'alloc_config', 'planner_param.yaml')
        results_dir = os.path.join(current_dir, 'results', curr_run, 'alloc_results.yaml')
    else:
        # If an input directory is provided, use it as the root directory.
        current_dir = str(run_dir)
        planner_yaml_dir = os.path.join(current_dir, 'alloc_config', 'planner_param.yaml')
        results_dir = os.path.join(current_dir, 'alloc_results.yaml')

    os.makedirs(os.path.dirname(results_dir), exist_ok=True)

    planner = TeamPlanner()
    if config_dicts:
        # Build the problem from in-memory dictionaries.
        planner.form_problem(curr_dir=current_dir, config_dicts=config_dicts)
    else:
        # Load the problem from files.
        if not os.path.exists(planner_yaml_dir):
            error_msg = f"File pattern error: 'planner_param.yaml' not found in '{os.path.dirname(planner_yaml_dir)}'"
            print(error_msg, file=sys.stderr)
            return "", False, None
        planner.form_problem(curr_dir=current_dir, param_file=planner_yaml_dir)

    planner.optimize()
    allc_data = planner.post_process(results_dir, planner_yaml_dir)

    return (results_dir, planner.flag_optimized, allc_data)

if __name__ == "__main__":
    tango()
