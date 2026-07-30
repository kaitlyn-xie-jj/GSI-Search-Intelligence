"""Small canonical scenario set for M6 policy smoke benchmarks."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from ..contracts import SearchTask
from ..search_space import SearchGrid
from .contracts import SearchBenchmarkScenario


def default_benchmark_scenarios() -> Tuple[SearchBenchmarkScenario, ...]:
    """Return four matched layouts spanning useful and misleading priors."""
    return (
        _rectangular_scenario(
            "correct-prior",
            prior_condition="correct",
            target_row=3,
            target_column=4,
            focus_cells=((3, 4), (3, 3), (2, 4)),
            focus_mass=0.75,
        ),
        _rectangular_scenario(
            "uniform-prior",
            prior_condition="uniform",
            target_row=3,
            target_column=4,
        ),
        _rectangular_scenario(
            "noisy-prior",
            prior_condition="noisy",
            target_row=2,
            target_column=3,
            focus_cells=((2, 3), (2, 2), (1, 3), (0, 4)),
            focus_mass=0.55,
        ),
        _rectangular_scenario(
            "misleading-prior",
            prior_condition="misleading",
            target_row=3,
            target_column=4,
            focus_cells=((0, 0), (0, 1), (1, 0)),
            focus_mass=0.75,
        ),
    )


def focused_grid_belief(
    grid: SearchGrid,
    focused_cell_ids: Iterable[str],
    focus_mass: float,
) -> Dict[str, float]:
    """Allocate a fixed mass to a semantic focus region and spread the rest."""
    if not 0 <= focus_mass <= 1:
        raise ValueError("focus_mass must be within [0, 1]")
    searchable_ids = tuple(cell.cell_id for cell in grid.searchable_cells)
    focused = tuple(dict.fromkeys(str(item) for item in focused_cell_ids))
    unknown = set(focused) - set(searchable_ids)
    if unknown:
        raise ValueError(f"focused cells are outside the search grid: {sorted(unknown)}")
    if not focused:
        probability = 1.0 / len(searchable_ids)
        return {cell_id: probability for cell_id in searchable_ids}
    remaining = tuple(cell_id for cell_id in searchable_ids if cell_id not in focused)
    if not remaining and focus_mass < 1:
        raise ValueError("focus_mass must be 1 when every cell is focused")
    belief = {
        cell_id: focus_mass / len(focused)
        for cell_id in focused
    }
    if remaining:
        belief.update({
            cell_id: (1.0 - focus_mass) / len(remaining)
            for cell_id in remaining
        })
    return belief


def _rectangular_scenario(
    scenario_id: str,
    *,
    prior_condition: str,
    target_row: int,
    target_column: int,
    focus_cells: Tuple[Tuple[int, int], ...] = (),
    focus_mass: float = 0.0,
) -> SearchBenchmarkScenario:
    area_id = f"benchmark-{scenario_id}"
    task = SearchTask.from_skill_params({
        "task_id": scenario_id,
        "area_token": area_id,
        "area": {
            "kind": "rectangle",
            "coords": [[0, 0], [100, 0], [100, 80], [0, 80]],
        },
        "target_token": "yellow-van",
        "max_viewpoints": 24,
        "conf_ge": 0.5,
    })
    grid = SearchGrid.from_task(task, resolution_m=20.0)
    target_cell = grid.cell(target_row, target_column)
    if target_cell is None:
        raise ValueError("default target index is outside the grid")
    focus_ids = tuple(
        grid.cell(row, column).cell_id
        for row, column in focus_cells
        if grid.cell(row, column) is not None
    )
    belief = (
        focused_grid_belief(grid, focus_ids, focus_mass)
        if focus_ids else grid.uniform_belief()
    )
    return SearchBenchmarkScenario(
        scenario_id=scenario_id,
        task=task,
        grid=grid,
        target_cell_id=target_cell.cell_id,
        initial_belief=belief,
        start_xy=(10.0, 10.0),
        prior_condition=prior_condition,
        metadata={
            "layout": "100m_x_80m_rectangle",
            "grid_resolution_m": 20.0,
            "prior_focus_mass": focus_mass,
        },
    )
