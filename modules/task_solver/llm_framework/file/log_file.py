
from datetime import datetime
import sys
import time
from .logger import setup_logger, LoggerLevel, _ANSI_COLOR_CODES
from .file import File


class Logger:

    def __init__(self, log_file_dir: str):
        """
        Initializes a new logger instance for a specific log file.
        
        :param log_file_dir: The full path to the log file.
        """
        self._logger = setup_logger(f"log_{id(self)}", LoggerLevel.DEBUG)
        self._clean_logger = setup_logger(f"clean_log_{id(self)}", LoggerLevel.DEBUG, clean_format=True)
        
        # Create the file object upon initialization
        self._file = File("log.md", log_file_dir)
        self._file.write("", mode="w") # Clear the file at the start of a run

    def log(self, content: str, level: str = "info", print_to_terminal: bool = True, 
            stream_output: bool = False, stream_delay: float = 0.07, clean_output: bool = True):
        """
        Formats a message based on the provided style and logs the content.
        
        :param content: The message content to be formatted and logged.
        :param level: The style to format the message. Supported styles: stage, action,
                      prompt, response, success, error, warning, info, debug.
        :param print_to_terminal: Whether to print to terminal
        :param stream_output: Whether to stream the output line by line
        :param stream_delay: Delay between each line when streaming (in seconds)
        :param clean_output: Whether to print only the message without log formatting
        """
        color_mapping = {
            "stage": '***\n# <span style="color: blue;">Current Stage: *{}*</span>\n',
            "action": '## <span style="color: purple;">Current Action: *{}*</span>\n',
            "prompt": '### <span style="color: grey ;">Prompt: </span>\n{}\n',
            "response": '### <span style="color: black;">Response: </span>\n{}\n',
            "success": '#### <span style="color: gold;">Success: {}</span>\n',
            "error": '#### <span style="color: red;">Error: </span>\n{}\n',
            "warning": '#### <span style="color: orange;">Warning: </span>\n{}\n',
            "info": '#### <span style="color: black;">info: </span>\n{}\n',
            "debug": '#### <span style="color: black;">debug: </span>\n{}\n',
        }

        # Get current time as a timestamp
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S:%f]")

        # Verify level is supported
        if level not in color_mapping:
            self._logger.error(f"Level {level} is not supported")
            return

        # Get the appropriate log action
        if clean_output:
            # Use clean logger for output without formatting
            log_action = {
                "stage": self._clean_logger.info,
                "action": self._clean_logger.debug,
                "prompt": self._clean_logger.debug,
                "response": self._clean_logger.info,
                "success": self._clean_logger.info,
                "error": self._clean_logger.error,
                "warning": self._clean_logger.warning,
                "info": self._clean_logger.info,
                "debug": self._clean_logger.debug,
            }.get(level, self._clean_logger.info)
        else:
            # Use regular logger with formatting
            log_action = {
                "stage": self._logger.info,
                "action": self._logger.debug,
                "prompt": self._logger.debug,
                "response": self._logger.info,
                "success": self._logger.info,
                "error": self._logger.error,
                "warning": self._logger.warning,
                "info": self._logger.info,
                "debug": self._logger.debug,
            }.get(level, self._logger.info)

        # Format content with timestamp
        content_with_timestamp = f"{timestamp}:{content}"

        if print_to_terminal:
            if clean_output:
                # Print directly to stdout without logger formatting but with colors
                if stream_output:
                    self._stream_clean_output(content_with_timestamp, stream_delay, level)
                else:
                    color, reset = self._get_level_color(level)
                    print(f"{color}{content_with_timestamp}{reset}")
            else:
                # Use logger with formatting
                if stream_output:
                    self._stream_log_output(content_with_timestamp, log_action, stream_delay)
                else:
                    log_action(content_with_timestamp)

        # Write to the file with color formatting
        self._file.write(color_mapping[level].format(f"{timestamp}:{content}"), mode="a")

    def _get_level_color(self, level: str):
        """
        Get the color code for a specific log level.
        
        :param level: The log level
        :return: Tuple of (color_code, reset_code)
        """
        level_color_mapping = {
            "stage": _ANSI_COLOR_CODES["BLUE"],
            "action": _ANSI_COLOR_CODES["BLUE"], 
            "prompt": _ANSI_COLOR_CODES["BLUE"],
            "response": _ANSI_COLOR_CODES["GREEN"],
            "success": _ANSI_COLOR_CODES["GREEN"],
            "error": _ANSI_COLOR_CODES["RED"],
            "warning": _ANSI_COLOR_CODES["YELLOW"],
            "info": _ANSI_COLOR_CODES["GREEN"],
            "debug": _ANSI_COLOR_CODES["BLUE"],
        }
        
        color = level_color_mapping.get(level, _ANSI_COLOR_CODES["GREEN"])
        reset = _ANSI_COLOR_CODES["RESET"]
        return color, reset

    def _stream_clean_output(self, content: str, delay: float, level: str = "info"):
        """
        Stream the output line by line without logger formatting but with colors.
        
        :param content: The content to be streamed
        :param delay: Delay between each line
        :param level: Log level to determine color
        """
        color, reset = self._get_level_color(level)
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Print with color formatting
            print(f"{color}{line}{reset}")
            
            # Add delay between lines (except for the last line)
            if i < len(lines) - 1:
                time.sleep(delay)
                
        # Ensure the output is flushed immediately
        sys.stdout.flush()

    def _stream_log_output(self, content: str, log_action, delay: float):
        """
        Stream the log output line by line with specified delay.
        
        :param content: The content to be streamed
        :param log_action: The logging function to use
        :param delay: Delay between each line
        """
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip():  # Only process non-empty lines
                log_action(line)
            elif i < len(lines) - 1:  # Print empty lines except the last one
                log_action("")
            
            # Add delay between lines (except for the last line)
            if i < len(lines) - 1:
                time.sleep(delay)
                
        # Ensure the output is flushed immediately
        sys.stdout.flush()

    # ==================== Convenience methods ====================
    
    def warning(self, content: str, print_to_terminal: bool = True, clean_output: bool = True):
        """
        Convenience method for logging warnings.
        
        :param content: Warning message content
        :param print_to_terminal: Whether to print to terminal
        :param clean_output: Whether to use clean output format
        """
        self.log(content, level="warning", print_to_terminal=print_to_terminal, clean_output=clean_output)
    
    def error(self, content: str, print_to_terminal: bool = True, clean_output: bool = True):
        """
        Convenience method for logging errors.
        
        :param content: Error message content
        :param print_to_terminal: Whether to print to terminal
        :param clean_output: Whether to use clean output format
        """
        self.log(content, level="error", print_to_terminal=print_to_terminal, clean_output=clean_output)
    
    def info(self, content: str, print_to_terminal: bool = True, clean_output: bool = True):
        """
        Convenience method for logging info.
        
        :param content: Info message content
        :param print_to_terminal: Whether to print to terminal
        :param clean_output: Whether to use clean output format
        """
        self.log(content, level="info", print_to_terminal=print_to_terminal, clean_output=clean_output)
    
    def debug(self, content: str, print_to_terminal: bool = True, clean_output: bool = True):
        """
        Convenience method for logging debug info.
        
        :param content: Debug message content
        :param print_to_terminal: Whether to print to terminal
        :param clean_output: Whether to use clean output format
        """
        self.log(content, level="debug", print_to_terminal=print_to_terminal, clean_output=clean_output)
    
    def success(self, content: str, print_to_terminal: bool = True, clean_output: bool = True):
        """
        Convenience method for logging success.
        
        :param content: Success message content
        :param print_to_terminal: Whether to print to terminal
        :param clean_output: Whether to use clean output format
        """
        self.log(content, level="success", print_to_terminal=print_to_terminal, clean_output=clean_output)
