
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

from modules.task_solver.llm_framework.llm import BaseLLM, model_manager


class QWEN(BaseLLM):
    def __init__(self, memorize: bool = False, stream_output: bool = False) -> None:
        self.api_base, self.key, self.model = model_manager.allocate(
            model_family="QWEN"
        )
        super().__init__(self.model, memorize, stream_output)
        self._client = AsyncOpenAI(api_key=self.key, base_url=self.api_base)

    @retry(
        stop=(stop_after_attempt(5) | stop_after_delay(500)),
        wait=wait_random_exponential(multiplier=1, max=60),
    )
    async def _ask_with_retry(self, temperature: float) -> str:
        """
        Helper function to perform the actual call to the GPT model with retry logic.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._memories,
                temperature=temperature,
                stream=self._stream_output,
            )

            if self._stream_output:
                collected_chunks = []
                collected_messages = []
                async for chunk in response:
                    collected_chunks.append(chunk)
                    choices = chunk.choices if hasattr(chunk, "choices") else []
                    if len(choices) > 0:
                        chunk_message = (
                            choices[0].delta if hasattr(choices[0], "delta") else {}
                        )
                        collected_messages.append(chunk_message)

                full_reply_content = "".join(
                    [
                        m.content
                        if hasattr(m, "content") and m.content is not None
                        else ""
                        for m in collected_messages
                    ]
                )
                self._response = full_reply_content
                return full_reply_content
            else:
                return response.choices[0].message.content

        except Exception as e:
            from modules.task_solver.llm_framework.file.log_file import logger

            logger.log(f"Error in _ask_with_retry: {e}", level="error")
            raise  # Re-raise exception to trigge


if __name__ == "__main__":
    import asyncio

    qwen = QWEN()

    response = asyncio.run(qwen.ask("Hello, who are you?"))
    print(response)
