
from abc import ABC, abstractmethod


class Context(ABC):
    @abstractmethod
    def save_to_file(self, filename):
        pass

    @abstractmethod
    def load_from_file(self, filename):
        pass

    @property
    def command(self):
        raise NotImplementedError
