
# import asyncio
# import traceback

# from openai import AsyncOpenAI
# from modules.task_solver.llm_framework.file import Logger
# from modules.task_solver.llm_framework.llm.llm import BaseLLM
# from modules.task_solver.llm_framework.llm.model_manager import model_manager
# from tenacity import (
#     retry,
#     stop_after_attempt,
#     wait_random_exponential,
#     stop_after_delay,
# )

# import httpx


# class GPT(BaseLLM):
#     """
#     A class to interact with OpenAI's GPT model.

#     This class handles requests to OpenAI's GPT models using an asynchronous client,
#     providing retry mechanisms and optional streaming support.

#     Args:
#         memorize (bool): Whether to store previous interactions for context in future requests.
#         stream_output (bool): Whether to receive partial outputs via streaming.
#     """

#     def __init__(
#             self, logger: Logger, memorize: bool = False, stream_output: bool = False, model_family: str = None, model_name_override: str = None
#     ) -> None:
#         """
#         Initializes the GPT class by allocating a model, obtaining the necessary API
#         credentials, and initializing the asynchronous client.

#         Args:
#             memorize (bool): Flag indicating if the class should store previous interactions.
#             stream_output (bool): Flag indicating if output should be streamed.
#             model_family (str, optional): The model family to use (e.g., "GPT", "GPT4o").
#                                    If None, falls back to a default family ("GPT").
#             model_name_override (str, optional): If provided, overrides the default model
#                                                  name for the chosen family.
#         """
#         self.logger = logger

#         model_family_to_use = model_family if model_family else "GPT"
#         self.api_base, self.key, self.model = model_manager.allocate(
#             model_family=model_family_to_use,
#             model_name_override=model_name_override
#         )
#         super().__init__(self.model, memorize, stream_output)
#         self._client = AsyncOpenAI(api_key=self.key, base_url=self.api_base)
#         # self._client = AsyncOpenAI(api_key=self.key, base_url=self.api_base, http_client=httpx_client)

#     @retry(
#         stop=(stop_after_attempt(5) | stop_after_delay(500)),
#         wait=wait_random_exponential(multiplier=1, max=60),
#         reraise=True,
#     )
#     async def _make_request(self, temperature: float) -> str:
#         """
#         Sends a request to the GPT model and retrieves the result, with optional retry logic.

#         This function handles both standard and streaming outputs. In streaming mode, it
#         collects all the chunks and assembles them into a complete response.

#         Args:
#             temperature (float): Controls the randomness of the output. Higher values produce more varied responses.

#         Returns:
#             str: The final content returned by the GPT model.

#         Raises:
#             Exception: If an error occurs during the API call, it will be logged and re-raised.
#         """
#         try:
#             response = await self._client.chat.completions.create(
#                 model=self._model,
#                 messages=[self._memories[-1]],
#                 temperature=temperature,
#                 stream=self._stream_output,
#             )

#             if self._stream_output:
#                 # Handle streaming output
#                 collected_chunks = []
#                 collected_messages = []
#                 async for chunk in response:
#                     collected_chunks.append(chunk)
#                     choices = chunk.choices if hasattr(chunk, "choices") else []
#                     if len(choices) > 0:
#                         chunk_message = (
#                             choices[0].delta if hasattr(choices[0], "delta") else {}
#                         )
#                         collected_messages.append(chunk_message)

#                 # Assemble full content from streaming chunks
#                 full_reply_content = "".join(
#                     [
#                         m.content
#                         if hasattr(m, "content") and m.content is not None
#                         else ""
#                         for m in collected_messages
#                     ]
#                 )
#                 return full_reply_content
#             else:
#                 # Return the first message's content if not streaming
#                 return response.choices[0].message.content
#         except Exception as e:
#             traceback.print_exc()
#             self.logger.log(f"Error in _make_request: {e}", level="error")
#             raise  # Re-raise the exception to trigger the retry logic

#     async def _retry_request_with_sleep(self, temperature: float) -> str:
#         """
#         Continuously retries the GPT request with a delay between attempts.

#         This method sleeps for 5 minutes between retries and continues until a successful
#         request is made. It logs each retry attempt.

#         Args:
#             temperature (float): The temperature parameter to control the response randomness.

#         Returns:
#             str: The final result returned by the GPT model after a successful request.
#         """

#         while True:
#             self.logger.log(
#                 "Sleeping for 5 minutes before retrying request...", level="info"
#             )
#             await asyncio.sleep(5 * 60)  # Sleep for 5 minutes

#             try:
#                 # Attempt to make the request again after sleep
#                 result = await self._make_request(temperature)
#                 return result  # Return result upon success
#             except Exception as e:
#                 self.logger.log(f"Request failed in sleep mode: {e}", level="error")
#                 continue  # Continue to retry if the request fails

#     async def _ask_with_retry(self, temperature: float) -> str:
#         """
#         A helper method to perform the GPT model request with retry logic.

#         If the maximum retry attempts (5) are exceeded, this method falls back to the
#         retry-with-sleep strategy, where the request is retried every 5 minutes.

#         Args:
#             temperature (float): The temperature parameter for controlling the response variability.

#         Returns:
#             str: The final content returned by the GPT model, after handling retries or sleep mode.
#         """

#         try:
#             # First attempt to make the request
#             return await self._make_request(temperature)
#         except Exception as re:
#             self.logger.log(f"Exceeded 5 retries, entering sleep mode: {re}", level="error")
#             # After retries are exhausted, switch to retry-with-sleep mode
#             return await self._retry_request_with_sleep(temperature)

#     async def __aenter__(self):
#         return self

#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         await self.close()

# if __name__ == "__main__":
#     gpt = GPT()
#     response = asyncio.run(gpt.ask("Hello, who are you?"))
#     print(response)


import asyncio
import traceback

from openai import AsyncOpenAI
from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.llm_framework.llm.llm import BaseLLM
from modules.task_solver.llm_framework.llm.model_manager import get_openrouter_extra_body, model_manager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    stop_after_delay,
)

import httpx

# proxies = {
#     "http://": "socks5://127.0.0.1:7890",
#     "https://": "socks5://127.0.0.1:7890",
# }
# httpx_client = httpx.AsyncClient(proxies=proxies)


class GPT(BaseLLM):
    """
    A class to interact with OpenAI's GPT model.

    This class handles requests to OpenAI's GPT models using an asynchronous client,
    providing retry mechanisms and optional streaming support.

    Args:
        memorize (bool): Whether to store previous interactions for context in future requests.
        stream_output (bool): Whether to receive partial outputs via streaming.
    """

    def __init__(
            self, logger: Logger, memorize: bool = False, stream_output: bool = False, model_family: str = None, model_name_override: str = None
    ) -> None:
        """
        Initializes the GPT class by allocating a model, obtaining the necessary API
        credentials, and initializing the asynchronous client.

        Args:
            memorize (bool): Flag indicating if the class should store previous interactions.
            stream_output (bool): Flag indicating if output should be streamed.
            model_family (str, optional): The model family to use (e.g., "GPT", "GPT4o").
                                   If None, falls back to a default family ("GPT").
            model_name_override (str, optional): If provided, over  rides the default model
                                                 name for the chosen family.
        """
        self.logger = logger

        model_family_to_use = model_family if model_family else "GPT"
        self.api_base, self.key, self.model = model_manager.allocate(
            model_family=model_family_to_use,
            model_name_override=model_name_override
        )
        super().__init__(self.model, memorize, stream_output)
        use_direct_local_client = (
            isinstance(self.api_base, str)
            and (
                self.api_base.startswith("http://localhost")
                or self.api_base.startswith("http://127.0.0.1")
            )
        )
        http_client = httpx.AsyncClient(trust_env=False) if use_direct_local_client else None
        self._client = AsyncOpenAI(api_key=self.key, base_url=self.api_base, http_client=http_client)
        # self._client = AsyncOpenAI(api_key=self.key, base_url=self.api_base, http_client=httpx_client)
        self._extra_body = get_openrouter_extra_body()

    @retry(
        stop=(stop_after_attempt(1) | stop_after_delay(500)),
        wait=wait_random_exponential(multiplier=1, max=60),
        reraise=True,
    )
    async def _make_request(self, temperature: float) -> str:
        """
        Sends a request to the GPT model and retrieves the result, with optional retry logic.

        This function handles both standard and streaming outputs. In streaming mode, it
        collects all the chunks and assembles them into a complete response.

        Args:
            temperature (float): Controls the randomness of the output. Higher values produce more varied responses.

        Returns:
            str: The final content returned by the GPT model.

        Raises:
            Exception: If an error occurs during the API call, it will be logged and re-raised.
        """
        try:
            request_kwargs = {
                "model": self._model,
                "messages": [self._memories[-1]],
                "temperature": temperature,
                "stream": self._stream_output,
            }
            extra_body = getattr(self, "_extra_body", None) or get_openrouter_extra_body()
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            response = await self._client.chat.completions.create(**request_kwargs)

            if self._stream_output:
                # Handle streaming output
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

                # Assemble full content from streaming chunks
                full_reply_content = "".join(
                    [
                        m.content
                        if hasattr(m, "content") and m.content is not None
                        else ""
                        for m in collected_messages
                    ]
                )
                return full_reply_content
            else:
                # Return the first message's content if not streaming
                return response.choices[0].message.content
        except Exception as e:
            traceback.print_exc()
            self.logger.log(f"Error in _make_request: {e}", level="error")
            raise  # Re-raise the exception to trigger the retry logic

    async def _retry_request_with_sleep(self, temperature: float) -> str:
        """
        Continuously retries the GPT request with a delay between attempts.

        This method sleeps for 5 minutes between retries and continues until a successful
        request is made. It logs each retry attempt.

        Args:
            temperature (float): The temperature parameter to control the response randomness.

        Returns:
            str: The final result returned by the GPT model after a successful request.
        """

        while True:
            self.logger.log(
                "Sleeping for 5 minutes before retrying request...", level="info"
            )
            await asyncio.sleep(5 * 60)  # Sleep for 5 minutes

            try:
                # Attempt to make the request again after sleep
                result = await self._make_request(temperature)
                return result  # Return result upon success
            except Exception as e:
                self.logger.log(f"Request failed in sleep mode: {e}", level="error")
                continue  # Continue to retry if the request fails

    async def _ask_with_retry(self, temperature: float) -> str:
        """
        A helper method to perform the GPT model request with retry logic.

        If the maximum retry attempts (5) are exceeded, this method falls back to the
        retry-with-sleep strategy, where the request is retried every 5 minutes.

        Args:
            temperature (float): The temperature parameter for controlling the response variability.

        Returns:
            str: The final content returned by the GPT model, after handling retries or sleep mode.
        """

        try:
            # First attempt to make the request
            return await self._make_request(temperature)
        except Exception as re:
            self.logger.log(f"Exceeded 5 retries, entering sleep mode: {re}", level="error")
            # After retries are exhausted, switch to retry-with-sleep mode
            return await self._retry_request_with_sleep(temperature)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

if __name__ == "__main__":
    gpt = GPT()
    response = asyncio.run(gpt.ask("Hello, who are you?"))
    print(response)