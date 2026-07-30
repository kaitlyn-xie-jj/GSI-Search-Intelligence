from typing import List, Dict, Any, Tuple
from modules.task_solver.sgi_planner.allocator.tango import generate_config_files, save_config_files, tango

def tango_allocator(config: Dict[str, Any], run_dir: str) -> str:
    # Generate and save configuration
    config_dicts = generate_config_files(config)
    save_config_files(config_dicts, save_dir=run_dir)
    
    # Run allocation algorithm
    scene_d, task_d, vehicle_d, planner_d = config_dicts
    in_memory_configs = {
        "planner": planner_d,
        "scene": scene_d,
        "task": task_d,
        "vehicle": vehicle_d
    }
    
    allocation_results_file, flag_optimized, allc_data = tango(run_dir=run_dir, config_dicts=in_memory_configs)
    
    return allocation_results_file, flag_optimized, allc_data