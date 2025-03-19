from typing import Optional
from huggingface_hub import repo_info
import traceback


def error_explainer(e: Exception) -> None:
    "Prints exception type, message, line number and full traceback"
    exception_type = type(e).__name__
    exception_message = str(e)
    exception_traceback = traceback.extract_tb(e.__traceback__)
    line_number = exception_traceback[-1].lineno

    print(f"Exception Type: {exception_type}")
    print(f"Exception Message: {exception_message}")
    print(f"Line Number: {line_number}")
    print("Full Traceback:")
    print("".join(traceback.format_tb(e.__traceback__)))


def hf_repo_exists(
    repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None
) -> bool:
    "Checks whether repo_id mentioned is available on huggingface"
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except Exception:
        return False
