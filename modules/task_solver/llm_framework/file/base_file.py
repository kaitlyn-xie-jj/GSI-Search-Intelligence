
from abc import abstractmethod


class BaseFile:
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, content):
        pass

    @property
    def message(self):
        pass

    @message.setter
    def message(self, value):
        pass
