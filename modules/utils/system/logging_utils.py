from typing import Optional, Any

from modules.config.system_config import config
from modules.task_solver.llm_framework.file import Logger 

def dlog(
    *args: Any,
    logger: Optional[Logger] = None,
    level: str = "info",
    sep: str = ' ',
    **kwargs: Any
):
    """
    A general logging and conditional print helper that behaves like print.

    Core behavior:
    - If a logger instance is provided, logs are always written to file.
    - Messages print to terminal only when config.enable_detailed_print is True.
    - Without a logger instance, this acts like print fully controlled by
      config.enable_detailed_print.

    Args:
        *args (Any): Any number of positional args, like print.
        logger (Optional[Logger]): Logger instance used for file logging, if provided.
        level (str): Log level, such as 'info', 'debug', or 'error'.
        sep (str): Separator used to join positional args.
        **kwargs (Any): Extra keyword args passed to logger.log.
    """
    # Join all positional args into one string message.
    message = sep.join(map(str, args))
    
    # Decide whether to print to terminal from global config.
    should_print_to_terminal = config.enable_detailed_print

    if logger and isinstance(logger, Logger):
        # If a Logger instance exists, call its log method.
        # Logs always go to file, while terminal printing follows should_print_to_terminal.
        logger.log(
            message,
            level=level,
            print_to_terminal=should_print_to_terminal,
            **kwargs
        )
    elif should_print_to_terminal:
        # Without a Logger instance, fall back to standard print, still controlled by config.
        print(message)
