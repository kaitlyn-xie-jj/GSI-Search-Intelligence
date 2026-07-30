
from abc import ABC, abstractmethod
from modules.config.system_config import config as system_config


class BaseLLM(ABC):
    """
    Base class for interacting with different LLM models.

    This class defines the basic methods and properties required for interacting
    with various LLM models from different manufacturers.

    Args:
        model (str): Model to use.
        memorize (bool): Whether to memorize conversation history.
        stream_output (bool): Whether to use streaming output.
    """

    def __init__(
            self, model: str, memorize: bool = False, stream_output: bool = False
    ) -> None:
        self._model = model
        self._memorize = memorize
        self._stream_output = stream_output
        self._memories = []  # Current memories
        self._response: str
        self.system_prompt = "You are a helpful assistant."

    def reset(self, system_prompt: str) -> None:
        if system_prompt:
            self.system_prompt = system_prompt
        self._memories = []
        # self._memories.append({"role": "system", "content": self.system_prompt})

    async def ask(self, prompt: str | list, temperature: float = None) -> str:
        """
        Asynchronously generate an answer from the model based on the given prompt.

        Args:
            prompt: The input prompt (string or message list).
            temperature: Sampling temperature. If None, reads from config `llm_temperature` (default 0.0).
        """
        if temperature is None:
            temperature = system_config.get_config("llm_temperature", 0.0)
        self._memories.append({"role": "user", "content": prompt})
        response = await self._ask_with_retry(temperature)
        if self._memorize:
            self._memories.append({"role": "assistant", "content": response})
            
        return response

    @abstractmethod
    async def _ask_with_retry(self, temperature: float) -> str:
        """
        Abstract method to be implemented by subclasses for the actual call to the model with retry logic.
        """

        raise NotImplementedError
