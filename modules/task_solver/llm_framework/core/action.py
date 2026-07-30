
import asyncio
import traceback
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_random_exponential

from modules.task_solver.llm_framework.file import Logger, setup_logger, LoggerLevel
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.llm.gpt import GPT
from modules.utils.system.logging_utils import dlog

class BaseNode(ABC):
    def __init__(self):
        self._logger = setup_logger(self.__class__.__name__, LoggerLevel.DEBUG)
        self.__next = None 
        self._renderer = None

    def __str__(self):
        return self.__class__.__name__

    @property
    def _next(self):
        return self.__next

    @_next.setter
    def _next(self, value):
        self.__next = value

    @abstractmethod
    async def run(self, auto_next: bool = True) -> str:
        pass


class ActionNode(BaseNode):
    def __init__(
            self, logger: Logger, 
            context: WorkflowContext, 
            next_text: str = "", 
            node_name: str = "", 
            llm: GPT = None, 
            model_family: str = None, 
            model_name_override: str = None
    ):
        super().__init__()
        self.logger = logger
        self.context = context
        self.__llm = llm if llm else GPT(
            logger=self.logger,
            model_family=model_family,
            model_name_override=model_name_override
        )
        self.prompt = None
        self.resp_template = None
        self._next_text = next_text  
        self._node_name = node_name  

    def __str__(self):
        if self._node_name:
            return self._node_name
        else:
            # return class name when node_name is not defined
            return super(ActionNode, ActionNode).__str__(self)

    def _build_prompt(self):
        pass

    async def run(self, auto_next: bool = True) -> str:
        self._build_prompt()
        self.logger.log(f"Action: {str(self)}", level="action")
        res = await self._run()
        if auto_next and self._next is not None:
            return await self._next.run()

    @retry(
        stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=10)
    )
    async def _run(self) -> str:
        try:
            if self.prompt is None:
                raise ValueError(f"Prompt is required for ActionNode: {self}")
            # Structured markers for offline token counting
            self.logger.log("<!-- LLM_PROMPT_START -->", level="info", print_to_terminal=False)
            dlog(f"Prompt:\n {self.prompt}", logger=self.logger, level="debug")
            self.logger.log("<!-- LLM_PROMPT_END -->", level="info", print_to_terminal=False)
            code = await self.__llm.ask(self.prompt)
            self.logger.log("<!-- LLM_RESPONSE_START -->", level="info", print_to_terminal=False)
            dlog(f"Response:\n {code}", logger=self.logger, level="info")
            self.logger.log("<!-- LLM_RESPONSE_END -->", level="info", print_to_terminal=False)
            code = await self._process_response(code)
            return code
        except Exception as e:
            tb = traceback.format_exc()
            self.logger.log(f"Error in {str(self)}: {e},\n {tb}", "error")
            raise Exception

    async def _process_response(self, content: str) -> str:
        return content


class ActionLinkedList(BaseNode):
    def __init__(self, name: str, head: BaseNode):
        super().__init__()
        self.head = head  # property is used
        self._name = name  # name of the structure

    def __str__(self):
        if self._head:
            return str(self._head)

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        if isinstance(value, BaseNode):
            self._head = value
            self._tail = value
        else:
            raise TypeError("head must be a BaseNode")

    @property
    def _next(self):
        return self._tail._next

    @_next.setter
    def _next(self, value):
        self._tail._next = value

    def add(self, action: "BaseNode"):
        if isinstance(action, BaseNode):
            self._tail._next = action
            self._tail = action
        else:
            raise ValueError("Value must be a BaseNode")

    async def run(self, **kwargs):
        return await self._head.run()

    async def run_internal_actions(self, start_node=None):
        current_node = self._head if start_node is None else start_node
        while current_node:
            await current_node.run(auto_next=False)
            current_node = current_node._next


if __name__ == "__main__":
    pass
