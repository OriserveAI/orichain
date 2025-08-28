from typing import Any, List, Dict, Union
from orichain import error_explainer


class Embed(object):
    """
    Synchronous Embed class to get embeddings from TogetherAI API.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize TogetherAI client and set up API key.

        Args:
            - api_key (str): TogetherAI API key
            - timeout (Timeout, optional): Request timeout. Default is 60.0
            - max_retries (int, optional): Number of retries for the request. Default is 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif kwds.get("timeout") and not isinstance(kwds.get("timeout"), (int, float)):
            raise TypeError(
                "Invalid 'timeout' type detected:",
                type(kwds.get("timeout")),
                ", Please enter valid timeout (in seconds) in either int or float.",
            )
        elif kwds.get("max_retries") and not isinstance(kwds.get("max_retries"), int):
            raise TypeError(
                "Invalid 'max_retries' type detected:,",
                type(kwds.get("max_retries")),
                ", Please enter a value that is 'int'",
            )
        else:
            pass

        # Initialize the TogetherAI client with provided parameters
        from together import Together

        self.client = Together(
            api_key=kwds.get("api_key"),
            timeout=kwds.get("timeout", 60.0),
            max_retries=kwds.get("max_retries", 2),
        )

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

            response = self.client.embeddings.create(input=text, model=model_name)
            embeddings = [sentence.embedding for sentence in response.data]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncEmbed(object):
    """
    Asynchronous Embed class to get embeddings from TogetherAI API.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize TogetherAI client and set up API key.

        Args:
            - api_key (str): TogetherAI API key
            - timeout (Timeout, optional): Request timeout. Default is 60.0
            - max_retries (int, optional): Number of retries for the request. Default is 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif kwds.get("timeout") and not isinstance(kwds.get("timeout"), (int, float)):
            raise TypeError(
                "Invalid 'timeout' type detected:",
                type(kwds.get("timeout")),
                ", Please enter valid timeout (in seconds) in either int or float.",
            )
        elif kwds.get("max_retries") and not isinstance(kwds.get("max_retries"), int):
            raise TypeError(
                "Invalid 'max_retries' type detected:,",
                type(kwds.get("max_retries")),
                ", Please enter a value that is 'int'",
            )
        else:
            pass

        # Initialize the TogetherAI client with provided parameters
        from together import AsyncTogether

        self.client = AsyncTogether(
            api_key=kwds.get("api_key"),
            timeout=kwds.get("timeout", 60.0),
            max_retries=kwds.get("max_retries", 2),
        )

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

            response = await self.client.embeddings.create(input=text, model=model_name)
            embeddings = [sentence.embedding for sentence in response.data]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
