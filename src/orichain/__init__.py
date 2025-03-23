from typing import Optional
from huggingface_hub import repo_info
import traceback
import logging

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Mapping of critical errors
CRITICAL_EXCEPTIONS = (MemoryError, SystemExit, KeyboardInterrupt, RuntimeError)


def error_explainer(e: Exception) -> None:
    """
    Logs exception details with severity level based on the error type.

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

    error_details = (
        f"Exception Type: {exception_type}\n"
        f"Exception Message: {exception_message}\n"
        f"Line Number: {line_number}\n"
        f"Full Traceback:\n{full_traceback}"
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
