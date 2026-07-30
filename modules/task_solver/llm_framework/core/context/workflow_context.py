
import argparse
import pickle
from modules.task_solver.llm_framework.file import File
from modules.utils.system.root import PathManager
from .context import Context

class WorkflowContext(Context):
    """
    A context object that holds the state for a single, specific workflow run.
    It is NOT a Singleton.
    """
    def __init__(self, path_manager: PathManager):
        """
        Initializes the context for a new workflow run.

        Args:
            path_manager: A PathManager instance that provides the root directories for this run.
        """
        self.path_manager = path_manager
        self.command_file = File(name="command.md", root=str(path_manager.workspace_root))
        self.feedbacks = []
        self.args = argparse.Namespace(root=str(path_manager.workspace_root))
        self._generated_codes = []
        self._generated_text = {}

    def save_to_file(self, file_path: str):
        """Saves the entire context object to a file using pickle."""
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'WorkflowContext':
        """
        Loads a context object from a pickle file.
        This is a class method because you call it before an instance exists.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    @property
    def command(self) -> str:
        return self.command_file.message

    @command.setter
    def command(self, value: str):
        self.command_file.message = value
