#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan validator performance profiling with yappi
Precisely locate function-level performance bottlenecks
"""

import asyncio
import sys
from pathlib import Path

# Add project root directory
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import yappi
from modules.plan_validator.plan_validator import PlanValidator


# Test plan
TEST_PLAN = """{
  "atomic_tasks": [
    {
      "task_id": "T1",
      "description": "Transport the foundation_base from Hotel-1 and place it at Street Segment-25.",
      "location": "Street Segment-25",
      "required_skills": [
        {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
        {"skill_name": "place<foundation_base>_on<UGV>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "navigate<Street Segment-25>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
        {"skill_name": "navigate<Street Segment-25>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "place<foundation_base>_on<ground>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1}
      ],
      "dependencies": []
    },
    {
      "task_id": "T2",
      "description": "Transport the surveillance_mast from Hotel-1 to Street Segment-25.",
      "location": "Street Segment-25",
      "required_skills": [
        {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "navigate<Hotel-1>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1},
        {"skill_name": "place<surveillance_mast>_on<UGV>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "navigate<Street Segment-25>", "assigned_robot_type": ["UGV"], "assigned_robot_count": 1}
      ],
      "dependencies": ["T1"]
    },
    {
      "task_id": "T3",
      "description": "Assemble the surveillance pole.",
      "location": "Street Segment-25",
      "required_skills": [
        {"skill_name": "navigate<Street Segment-25>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1},
        {"skill_name": "place<surveillance_mast>_on<foundation_base>", "assigned_robot_type": ["Humanoid"], "assigned_robot_count": 1}
      ],
      "dependencies": ["T2"]
    }
  ],
  "meta": {"description": "Transport and assemble surveillance pole"}
}"""


async def run_validation():
    """Run validation task"""
    validator = PlanValidator(task_id="cybertown:scenario_1:g_100")
    result = await validator.validate_plan(TEST_PLAN)
    return result


def main():
    """Main function"""
    print("="*80)
    print("🔬 Performance profiling with yappi")
    print("="*80)
    
    # View current configuration
    from modules.config.system_config import config
    print(f"\n📋 Current configuration:")
    print(f"  - fine_grained_simulation: {config.fine_grained_simulation}")
    print(f"  - simulate_time_delay: {config.simulate_time_delay}")
    print(f"  - enable_visualization: {config.enable_visualization}")
    print(f"  - enable_new_case_generation: {config.enable_new_case_generation}")
    
    print(f"\n▶️  Starting yappi profiling...")
    
    # Setup yappi
    yappi.set_clock_type("wall")  # Use wall time instead of CPU time
    yappi.start()
    
    # Run validation
    result = asyncio.run(run_validation())
    
    # Stop yappi
    yappi.stop()
    
    print(f"\n✅ Validation complete!")
    print(f"   Validation result: {'Passed✓' if result['valid'] else 'Failed✗'}")
    print(f"   Reward score: {result['reward']}")
    print(f"   Total time: {result['validation_time']:.2f}s")
    
    # Get function statistics
    func_stats = yappi.get_func_stats()
    
    # Save detailed report
    output_dir = project_root / "plan_validation_results"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Sort by Total time
    print(f"\n" + "="*80)
    print("📊 Performance Report - Sorted by Total Time (Top 30)")
    print("="*80)
    print(f"{'Function Name':<60} {'Calls':>8} {'Total(s)':>10} {'Per Call(ms)':>10}")
    print("-"*80)
    
    func_stats.sort("ttot", "desc")
    top_functions = []
    
    for i, stat in enumerate(func_stats[:30]):
        func_name = stat.name
        if len(func_name) > 60:
            func_name = "..." + func_name[-57:]
        
        ncall = stat.ncall
        ttot = stat.ttot
        tavg = stat.tavg * 1000  # Convert to milliseconds
        
        print(f"{func_name:<60} {ncall:8d} {ttot:10.3f} {tavg:10.3f}")
        
        top_functions.append({
            "rank": i + 1,
            "function": stat.name,
            "module": stat.module,
            "ncall": ncall,
            "ttot": ttot,
            "tavg": tavg,
            "tsub": stat.tsub
        })
    
    # 2. Save full report to file
    report_path = output_dir / "yappi_profile_by_ttot.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("YAPPI Performance Report - Sorted by Total Time\n")
        f.write("="*100 + "\n")
        f.write(f"{'Function Name':<70} {'Calls':>10} {'Total(s)':>10} {'Per Call(ms)':>10}\n")
        f.write("-"*100 + "\n")
        
        func_stats.sort("ttot", "desc")
        for stat in func_stats:
            f.write(f"{stat.name:<70} {stat.ncall:10d} {stat.ttot:10.3f} {stat.tavg*1000:10.3f}\n")
    
    print(f"\n💾 Full report saved to: {report_path}")
    
    # 3. Sorted by Per-Call Time (find slow functions)
    print(f"\n" + "="*80)
    print("📊 Performance Report - Sorted by Per-Call Time (Top 20)")
    print("="*80)
    print(f"{'Function Name':<60} {'Calls':>8} {'Total(s)':>10} {'Per Call(ms)':>10}")
    print("-"*80)
    
    func_stats.sort("tavg", "desc")
    slow_functions = []
    
    for i, stat in enumerate(func_stats[:20]):
        # Filter out functions with too few Calls (may be random factors)
        if stat.ncall < 2:
            continue
            
        func_name = stat.name
        if len(func_name) > 60:
            func_name = "..." + func_name[-57:]
        
        ncall = stat.ncall
        ttot = stat.ttot
        tavg = stat.tavg * 1000
        
        print(f"{func_name:<60} {ncall:8d} {ttot:10.3f} {tavg:10.3f}")
        
        slow_functions.append({
            "rank": i + 1,
            "function": stat.name,
            "ncall": ncall,
            "ttot": ttot,
            "tavg": tavg
        })
    
    # 4. Save per-call time report
    report_path2 = output_dir / "yappi_profile_by_tavg.txt"
    with open(report_path2, 'w', encoding='utf-8') as f:
        f.write("YAPPI Performance Report - Sorted by Per-Call Time\n")
        f.write("="*100 + "\n")
        f.write(f"{'Function Name':<70} {'Calls':>10} {'Total(s)':>10} {'Per Call(ms)':>10}\n")
        f.write("-"*100 + "\n")
        
        func_stats.sort("tavg", "desc")
        for stat in func_stats:
            if stat.ncall >= 2:  # Only show functions with Calls>=2
                f.write(f"{stat.name:<70} {stat.ncall:10d} {stat.ttot:10.3f} {stat.tavg*1000:10.3f}\n")
    
    print(f"\n💾 Per-call time report saved to: {report_path2}")
    
    # 5. Generate call graph (pstat format)
    pstat_path = output_dir / "yappi_profile.pstat"
    func_stats.save(str(pstat_path), type='pstat')
    print(f"💾 pstat format saved to: {pstat_path}")
    print(f"   Visualization: snakeviz {pstat_path}")
    
    # 6. Generate callgrind format (viewable with kcachegrind)
    callgrind_path = output_dir / "yappi_profile.callgrind"
    func_stats.save(str(callgrind_path), type='callgrind')
    print(f"💾 callgrind format saved to: {callgrind_path}")
    
    # 7. Save JSON format summary
    import json
    summary_path = output_dir / "yappi_summary.json"
    summary = {
        "validation_result": {
            "valid": result['valid'],
            "reward": result['reward'],
            "total_time": result['validation_time']
        },
        "config": {
            "fine_grained_simulation": config.fine_grained_simulation,
            "simulate_time_delay": config.simulate_time_delay,
            "enable_visualization": config.enable_visualization,
            "enable_new_case_generation": config.enable_new_case_generation
        },
        "top_functions_by_total_time": top_functions[:10],
        "slow_functions_by_avg_time": slow_functions[:10]
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"💾 JSON summary saved to: {summary_path}")
    
    # 8. Key Module Time Analysis
    print(f"\n" + "="*80)
    print("🎯 Key Module Time Analysis")
    print("="*80)
    
    # Calculate time for each module
    module_stats = {}
    func_stats.sort("ttot", "desc")
    
    for stat in func_stats:
        module = stat.module
        if module not in module_stats:
            module_stats[module] = {"ttot": 0, "ncall": 0, "functions": []}
        
        module_stats[module]["ttot"] += stat.ttot
        module_stats[module]["ncall"] += stat.ncall
        module_stats[module]["functions"].append(stat.name)
    
    # Sort modules by Total time
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]["ttot"], reverse=True)
    
    print(f"{'Module Name':<50} {'Total(s)':>12} {'Calls':>12}")
    print("-"*80)
    
    for module, stats in sorted_modules[:20]:
        if stats["ttot"] < 0.01:  # Ignore modules with too little time
            continue
        module_name = module if len(module) <= 50 else "..." + module[-47:]
        print(f"{module_name:<50} {stats['ttot']:12.3f} {stats['ncall']:12d}")
    
    # 9. Performance Optimization Suggestions
    print(f"\n" + "="*80)
    print("💡 Performance Optimization Suggestions")
    print("="*80)
    
    # Analyze top functions to find optimization points
    func_stats.sort("ttot", "desc")
    
    suggestions = []
    
    for stat in func_stats[:20]:
        func_name = stat.name
        ttot = stat.ttot
        
        # asyncio.sleep related
        if 'sleep' in func_name.lower() and ttot > 0.5:
            suggestions.append(f"⚠️  Heavy sleep calls detected: {func_name} took {ttot:.2f}s")
            suggestions.append(f"    → Suggest setting simulate_time_delay=false")
        
        # Motion simulation related
        if 'simulate' in func_name.lower() and 'motion' in func_name.lower() and ttot > 0.5:
            suggestions.append(f"⚠️  Motion simulation time: {func_name} took {ttot:.2f}s")
            suggestions.append(f"    → Suggest setting fine_grained_simulation=false")
        
        # Path planning related
        if 'plan_path' in func_name.lower() and ttot > 0.5:
            suggestions.append(f"⚠️  Path planning time: {func_name} took {ttot:.2f}s")
        
        # Gurobi solver related
        if 'gurobi' in func_name.lower() or 'optimize' in func_name.lower():
            if ttot > 0.5:
                suggestions.append(f"⚠️  Optimization solver time: {func_name} took {ttot:.2f}s")
                suggestions.append(f"    → Consider setting solver time limit")
    
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("✓ No obvious performance bottlenecks found")
    
    print(f"\n" + "="*80)
    print("📝 Analysis complete! Check the following files for details:")
    print(f"  - By total time: {report_path}")
    print(f"  - By per-call time: {report_path2}")
    print(f"  - Visualization: snakeviz {pstat_path}")
    print(f"  - JSON summary: {summary_path}")
    print("="*80)
    
    # Clean up yappi
    yappi.clear_stats()


if __name__ == "__main__":
    main()
