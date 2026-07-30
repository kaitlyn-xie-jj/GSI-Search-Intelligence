
import datetime
from pathlib import Path
import argparse # Assuming you use argparse

def get_project_root() -> Path:
    """Search upwards to find the project root directory."""
    current_path = Path.cwd()
    while True:
        # Simplified the condition for clarity
        if (current_path / ".git").exists() or (current_path / ".project_root").exists():
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            return Path.cwd() # Fallback to current dir
        current_path = parent_path

class PathManager:
    """
    Manages all directory paths for a specific, single execution run.
    """
    def __init__(self,
                 task_name: str = '',
                 base_results_dir: str = "results",
                 formatted_date: str = '',
                 workspace_root_override: Path = None):
        """
        Initializes and creates all necessary paths for the run.

        Args:
            task_name: The name of the task, used for creating the output folder.
            base_results_dir: The root directory where all results are stored.
            formatted_date: Time-based or custom string for uniqueness.
            workspace_root_override: If provided, overrides default workspace root.
        """
        self.project_root: Path = get_project_root()
        
        if workspace_root_override is not None:
            # Use the provided run_dir directly.
            self.workspace_root: Path = Path(workspace_root_override)
        else:
            self.workspace_root: Path = (
                self.project_root
                / base_results_dir
                / task_name
                / formatted_date
            )
        
        # self.data_root: Path = self.workspace_root / "data"
        
        # Ensure the directories exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        # self.data_root.mkdir(exist_ok=True)
