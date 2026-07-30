
import os
import shutil
from enum import Enum
from pathlib import Path

from .base_file import BaseFile


class FileStatus(Enum):
    NOT_WRITTEN = 0
    NOT_TESTED = 1
    TESTED_FAIL = 2
    TESTED_PASS = 3


class File(BaseFile):
    """
    A simple file object that represents a file on the filesystem.
    It is responsible for reading and writing its own content.
    It does NOT handle logging; the calling context is responsible for that.
    """
    def __init__(self, name: str, root: str, message: str = ""):
        """
        Initializes a File object.

        Args:
            name: The name of the file (e.g., "flow.md").
            root: The absolute path to the directory where the file resides, as a string.
            message: Initial content for the file.
        """
        # Convert the incoming string path to a Path object for robust internal use.
        self.root = Path(root)
        self._name = name
        self._message = message

    @property
    def file_path(self) -> Path:
        """Returns the full, absolute path to the file as a Path object."""
        return self.root / self._name

    @property
    def message(self) -> str:
        """
        The content of the file. Lazily reads from disk if not already in memory.
        Raises FileNotFoundError if the file does not exist on disk.
        """
        if not self._message:
            # Check if the file exists before trying to read
            if self.file_path.is_file():
                self._message = self.read()
            else:
                # If file doesn't exist, message remains empty
                self._message = ""
        return self._message

    @message.setter
    def message(self, content: str):
        """Sets the content of the file and writes it to disk."""
        self._message = content
        self.write(content)

    def read(self) -> str:
        """
        Reads the file from disk and returns its content.
        Raises FileNotFoundError if the file does not exist.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, content: str, mode: str = "w") -> None:
        """
        Writes content to the file. Creates the root directory if it doesn't exist.
        Raises exceptions (e.g., PermissionError) on failure.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        
        with open(self.file_path, mode, encoding="utf-8") as f:
            f.write(content)

    def copy(self, new_root: str, new_name: str = "") -> 'File':
        """
        Creates a copy of this file at a new location.

        Args:
            new_root: The new directory path, as a string.
            new_name: Optional new name for the file.
        """
        new_name = new_name if new_name else self._name
        # The constructor will correctly handle the new_root string.
        new_file = File(name=new_name, root=new_root)
        new_file.message = self.message 
        return new_file
