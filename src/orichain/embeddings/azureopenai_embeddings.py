from typing import Any, List, Dict, Union
from asyncio import gather
import traceback


class Embed(object):
    def __init__(self, **kwds: Any) -> None:
        """
        Initialize Azure OpenAI client and set up API key.

        Args:
            - api_key (str): Azure OpenAI API key
        """
        from httpx import Timeout

        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("azure_endpoint"):
            raise KeyError("Required `azure_endpoint` not found")
        elif not kwds.get("api_version"):
            raise KeyError("Required `api_version` not found")
        elif kwds.get("timeout") and not isinstance(kwds.get("timeout"), Timeout):
            raise TypeError(
                "Invalid 'timeout' type detected:",
                type(kwds.get("timeout")),
                ", Please enter valid timeout using:\n'from httpx import Timeout'",
            )
        elif kwds.get("max_retries") and not isinstance(kwds.get("max_retries"), int):
            raise TypeError(
                "Invalid 'max_retries' type detected:,",
                type(kwds.get("max_retries")),
                ", Please enter a value that is 'int'",
            )
        else:
            pass

        from openai import AsyncAzureOpenAI
        import tiktoken

        self.client = AsyncAzureOpenAI(
            azure_endpoint=kwds.get("azure_endpoint"),
            api_key=kwds.get("api_key"),
            api_version=kwds.get("api_version"),
            timeout=kwds.get("timeout")
            or Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            max_retries=kwds.get("max_retries") or 2,
        )
        self.tiktoken = tiktoken

    async def __call__(
        self, text: Union[str, List[str]], model_name: str, **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        - Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            **kwargs: Additional keyword arguments for the embedding API

        - Returns:
            - Union[List[float], List[List[float]], Dict[str, Any]]:
            - Embeddings or error information
        """
        try:
            if isinstance(text, str):
                text = [text]

            token_counts = await gather(
                *[
                    self.num_tokens_from_string(sentence, model_name)
                    for sentence in text
                ]
            )

            max_tokens = max(token_counts)

            if max_tokens > 8192:
                return {
                    "error": 400,
                    "reason": f"Embedding model maximum context length is 8192 tokens, \
                            however you requested {max_tokens} tokens. Please reduce text size",
                }

            response = await self.client.embeddings.create(input=text, model=model_name)
            embeddings = []
            for sentence in response.data:
                embeddings.append(sentence.embedding)

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}

    async def num_tokens_from_string(self, string: str, model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = self.tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
