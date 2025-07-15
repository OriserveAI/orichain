from typing import Any, List, Dict, Union

from asyncio import gather
from concurrent.futures import ThreadPoolExecutor

from orichain import error_explainer


class Embed(object):
    """
    Synchronous Embed class to get embeddings from OpenAI API.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize OpenAI client and set up API key.

        Args:
            - api_key (str): OpenAI API key
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default is 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """
        from httpx import Timeout

        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
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

        from openai import OpenAI
        import tiktoken

        self.client = OpenAI(
            api_key=kwds.get("api_key"),
            timeout=kwds.get("timeout")
            or Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            max_retries=kwds.get("max_retries") or 2,
        )
        self.tiktoken = tiktoken

    def __call__(
        self, text: Union[str, List[str]], model_name: str, **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
        """
        try:
            if isinstance(text, str):
                text = [text]

            with ThreadPoolExecutor() as executor:
                token_counts = list(
                    executor.map(
                        lambda sentence: self.num_tokens_from_string(
                            sentence, model_name
                        ),
                        text,
                    )
                )

            max_tokens = max(token_counts)

            if max_tokens > 8192:
                return {
                    "error": 400,
                    "reason": f"Embedding model maximum context length is 8192 tokens, \
                            however you requested {max_tokens} tokens. Please reduce text size",
                }

            response = self.client.embeddings.create(input=text, model=model_name)
            embeddings = [sentence.embedding for sentence in response.data]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def num_tokens_from_string(self, string: str, model_name: str) -> int:
        """Returns the number of tokens in a text string.

        Args:
            string (str): Input text
            model_name (str): Name of the embedding model

        Returns:
            (int): Number of tokens in the text string
        """
        encoding = self.tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


class AsyncEmbed(object):
    """
    Asynchronous Embed class to get embeddings from OpenAI API.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize OpenAI client and set up API key.

        Args:
            - api_key (str): OpenAI API key
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default is 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """
        from httpx import Timeout

        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
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

        from openai import AsyncOpenAI
        import tiktoken

        self.client = AsyncOpenAI(
            api_key=kwds.get("api_key"),
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

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
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
            embeddings = [sentence.embedding for sentence in response.data]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def num_tokens_from_string(self, string: str, model_name: str) -> int:
        """Returns the number of tokens in a text string.
        Args:

            string (str): Input text
            model_name (str): Name of the embedding model

        Returns:
            (int): Number of tokens in the text string
        """
        encoding = self.tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
