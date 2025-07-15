from typing import Optional
from huggingface_hub import repo_info
import traceback
import logging
import sys

# Configure logging with colored output
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Mapping of critical errors
CRITICAL_EXCEPTIONS = (
    MemoryError,
    SystemExit,
    KeyboardInterrupt,
    RuntimeError,
    ImportError,
)


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Custom formatter that supports colored output
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Colors.GRAY
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Colors.RESET,
        logging.INFO: "%(asctime)s - %(levelname)s - %(message)s",
        logging.WARNING: Colors.YELLOW
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Colors.RESET,
        logging.ERROR: Colors.RED
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Colors.RESET,
        logging.CRITICAL: Colors.BOLD
        + Colors.RED
        + "%(asctime)s - %(levelname)s - %(message)s"
        + Colors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Apply the colored formatter to the console handler
colored_formatter = ColoredFormatter()
console_handler.setFormatter(colored_formatter)
logger.addHandler(console_handler)


def error_explainer(e: Exception) -> None:
    """
    Logs exception details with severity level and color coding based on the error type.

    Args:
        e (Exception): The exception object.
    """
    exception_type = type(e).__name__
    exception_message = str(e)

    # Extract traceback safely
    if e.__traceback__:
        exception_traceback = traceback.extract_tb(e.__traceback__)
        line_number = (
            exception_traceback[-1].lineno if exception_traceback else "Unknown"
        )
        full_traceback = "".join(traceback.format_tb(e.__traceback__))
    else:
        line_number = "Unknown"
        full_traceback = "No traceback available"

    # Colorize specific parts of the error message
    error_details = (
        f"{Colors.BOLD}Exception Type:{Colors.RESET} {Colors.MAGENTA}{exception_type}{Colors.RESET}\n"
        f"{Colors.BOLD}Exception Message:{Colors.RESET} {exception_message}\n"
        f"{Colors.BOLD}Line Number:{Colors.RESET} {Colors.CYAN}{line_number}{Colors.RESET}\n"
        f"{Colors.BOLD}Full Traceback:{Colors.RESET}\n{Colors.GRAY}{full_traceback}{Colors.RESET}"
    )

    # Decide log level based on exception type
    log_level = (
        logging.CRITICAL if isinstance(e, CRITICAL_EXCEPTIONS) else logging.ERROR
    )

    logging.log(log_level, error_details)


def hf_repo_exists(
    repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None
) -> bool:
    """Checks whether repo_id mentioned is available on huggingface

    Args:
        repo_id (str): Huggingface repo id
        repo_type (str): Type of repo
        token (str): Huggingface API token

    Returns:
        bool: True if repo exists, False otherwise
    """
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except Exception:
        return False
