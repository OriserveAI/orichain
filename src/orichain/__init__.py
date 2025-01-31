from typing import List, Dict, Optional, Union
from huggingface_hub import repo_info


async def validate_gen_response_request(
    user_message,
    metadata,
    prev_pairs,
    prev_chunks,
) -> Union[Dict, None]:
    """This fuction does a basic validation check of all the data type received in request
    If more validation checks are required in the brand, do it in utils.py"""

    # user_message check
    if not isinstance(user_message, str):
        return {"error": 400, "reason": "user_message is not string"}

    # metadata check
    if metadata and not isinstance(metadata, Dict):
        return {"error": 400, "reason": "metadata is not dict"}

    # prev_pairs check
    if isinstance(prev_pairs, List):
        role = "user"
        for dialogue in prev_pairs:
            if not isinstance(dialogue, Dict):
                return {
                    "error": 400,
                    "reason": "all elements in prev_pairs are not dict",
                }

            if dialogue.get("role"):
                if dialogue.get("role") != role:
                    return {
                        "error": 400,
                        "reason": "dicts in prev_pairs are not arranged correctly",
                    }
            else:
                return {
                    "error": 400,
                    "reason": "dict in prev_pairs does not have 'role' key present",
                }

            if not isinstance(dialogue.get("content"), str):
                return {
                    "error": 400,
                    "reason": "all messages in 'content' key are not string",
                }
            role = "assistant" if role == "user" else "user"

    else:
        if prev_pairs:
            return {"error": 400, "reason": "prev_pairs is not list"}

    # prev_chunks check
    if isinstance(prev_chunks, List):
        for chunk in prev_chunks:
            if not isinstance(chunk, str):
                return {
                    "error": 400,
                    "reason": "all elements in prev_chunks are not string",
                }
    else:
        if prev_chunks:
            return {"error": 400, "reason": "prev_chunks is not list"}

    return None


async def validate_result_request(
    user_message,
    bot_message,
    intent,
    brand_entity,
    universal_ner,
) -> Union[Dict, None]:
    # user_message check
    if not isinstance(user_message, str):
        return {"error": 400, "reason": "user_message is not string"}

    # bot_message check
    if not isinstance(bot_message, str):
        return {"error": 400, "reason": "bot_message is not string"}

    # intent check
    if intent and not isinstance(intent, Dict):
        return {"error": 400, "reason": "intent is not dict"}

    # brand_entity check
    if brand_entity and not isinstance(brand_entity, Dict):
        return {"error": 400, "reason": "brand_entity is not dict"}

    # intent check
    if universal_ner and not isinstance(universal_ner, Dict):
        return {"error": 400, "reason": "universal_ner is not dict"}

    return None


def hf_repo_exists(
    repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None
) -> bool:
    "Checks whether repo_id mentioned is available on huggingface"
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except Exception:
        return False
