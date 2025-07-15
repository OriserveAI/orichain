from typing import (
    Any,
    Dict,
    List,
    Union,
)
from orichain import error_explainer


class Embed(object):
    """
    Synchronous Embed class to get embeddings from Vertex AI.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Google client for Vertex AI and set up API key.

        Args:
            - api_key (str): Vertex AI API key
            - credentials (google.auth.credentials.Credentials): The credentials to use for authentication when calling the Vertex AI APIs.
            - project (str): The Google Cloud project ID to use for quota.
            - location (str): The location to send API requests to (for example, us-central1).
            - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
            - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from google.genai import Client, types
        from google.auth.credentials import Credentials
        from google.genai.client import DebugConfig

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("project"):
            raise KeyError("Required `project` not found")
        elif not kwds.get("location"):
            raise KeyError("Required `location` not found")
        elif kwds.get("credentials") and not isinstance(
            kwds.get("credentials"), Credentials
        ):
            raise TypeError(
                "Invalid 'credentials' type detected:",
                type(kwds.get("credentials")),
                ", Please enter valid credentials using:\n'from google.auth.credentials import Credentials'",
            )
        elif kwds.get("http_options") and not isinstance(
            kwds.get("http_options"), types.HttpOptions
        ):
            raise TypeError(
                "Invalid 'http_options' type detected:",
                type(kwds.get("http_options")),
                ", Please enter valid http_options using:\n'from google.genai.types import HttpOptions'",
            )
        elif kwds.get("debug_config") and not isinstance(
            kwds.get("debug_config"), DebugConfig
        ):
            raise TypeError(
                "Invalid 'debug_config' type detected:",
                type(kwds.get("debug_config")),
                ", Please enter valid debug_config using:\n'from google.genai.client import DebugConfig'",
            )
        else:
            pass

        # Initialize the Google client with provided parameters
        self.client: Client = Client(
            vertexai=True,
            api_key=kwds.get("api_key"),
            credentials=kwds.get("credentials"),
            project=kwds.get("project"),
            location=kwds.get("location"),
            http_options=kwds.get("http_options", types.HttpOptions(timeout=7000)),
            debug_config=kwds.get("debug_config"),
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
            response = self.client.models.embed_content(
                model=model_name,
                contents=text if isinstance(text, List) else [text],
                config=kwds.get("config"),
            )

            embeddings = [embedding.values for embedding in response.embeddings]
            if len(embeddings) == 1:
                embeddings = embeddings[-1]

            return embeddings

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}


class AsyncEmbed(object):
    """
    Asynchronous wrapper for Google's API client for Vertex AI.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Google client for Vertex AI and set up API key.

        Args:
            - api_key (str): Vertex AI API key
            - credentials (google.auth.credentials.Credentials): The credentials to use for authentication when calling the Vertex AI APIs.
            - project (str): The Google Cloud project ID to use for quota.
            - location (str): The location to send API requests to (for example, us-central1).
            - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
            - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from google.genai import Client, types
        from google.auth.credentials import Credentials
        from google.genai.client import DebugConfig

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("project"):
            raise KeyError("Required `project` not found")
        elif not kwds.get("location"):
            raise KeyError("Required `location` not found")
        elif kwds.get("credentials") and not isinstance(
            kwds.get("credentials"), Credentials
        ):
            raise TypeError(
                "Invalid 'credentials' type detected:",
                type(kwds.get("credentials")),
                ", Please enter valid credentials using:\n'from google.auth.credentials import Credentials'",
            )
        elif kwds.get("http_options") and not isinstance(
            kwds.get("http_options"), types.HttpOptions
        ):
            raise TypeError(
                "Invalid 'http_options' type detected:",
                type(kwds.get("http_options")),
                ", Please enter valid http_options using:\n'from google.genai.types import HttpOptions'",
            )
        elif kwds.get("debug_config") and not isinstance(
            kwds.get("debug_config"), DebugConfig
        ):
            raise TypeError(
                "Invalid 'debug_config' type detected:",
                type(kwds.get("debug_config")),
                ", Please enter valid debug_config using:\n'from google.genai.client import DebugConfig'",
            )
        else:
            pass

        # Initialize the Google client with provided parameters
        self.client: Client = Client(
            vertexai=True,
            api_key=kwds.get("api_key"),
            credentials=kwds.get("credentials"),
            project=kwds.get("project"),
            location=kwds.get("location"),
            http_options=kwds.get("http_options", types.HttpOptions(timeout=7000)),
            debug_config=kwds.get("debug_config"),
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
            response = await self.client.aio.models.embed_content(
                model=model_name,
                contents=text if isinstance(text, List) else [text],
                config=kwds.get("config"),
            )

            embeddings = [embedding.values for embedding in response.embeddings]
            if len(embeddings) == 1:
                embeddings = embeddings[-1]

            return embeddings

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}
