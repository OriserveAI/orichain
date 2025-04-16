from typing import Any, List, Dict, Union
from orichain.embeddings import (
    openai_embeddings,
    awsbedrock_embeddings,
    stransformers_embeddings,
    azureopenai_embeddings,
)
import warnings
from orichain import hf_repo_exists

DEFUALT_EMBEDDING_MODEL = "text-embedding-ada-002"
MODEL_CLASS = {
    "text-embedding-ada-002": "OpenAI",
    "text-embedding-3-large": "OpenAI",
    "text-embedding-3-small": "OpenAI",
    "amazon.titan-embed-text-v1": "AWSBedrock",
    "amazon.titan-embed-text-v2:0": "AWSBedrock",
    "cohere.embed-english-v3": "AWSBedrock",
    "cohere.embed-multilingual-v3": "AWSBedrock",
}


class EmbeddingModel(object):
    """Synchronus Base class for embedding generation
    Default embedding model that will be used is `text-embedding-ada-002`"""

    default_model = DEFUALT_EMBEDDING_MODEL
    model_class = MODEL_CLASS

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Embedding Models class with the required parameters.
        Args:
            - model_name (str, optional): Name of the model to be used. Default: "gpt-4o-mini"

            ### Authentication parameters by provider:

            #### OpenAI models
            - api_key (str): OpenAI API key.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

            #### AWS Bedrock models
            - aws_access_key (str): AWS access key.
            - aws_secret_key (str): AWS secret key.
            - aws_region (str): AWS region name.
            - config (Config, optional):
                - connect_timeout (float or int, optional): The time in seconds till a timeout exception is
                thrown when attempting to make a connection. Default: 60
                - read_timeout: (float or int, optional): The time in seconds till a timeout exception is
                thrown when attempting to read from a connection. Default: 60
                - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                - retries (Dict, optional):
                    - total_max_attempts: Number of retries for the request. Default: 2

            #### Sentence Transformers models
            - model_download_path (str, optional): Path to download the model. Default: "/home/ubuntu/projects/models/embedding_models"
            - device (str, optional): Device to run the model. Default: "cpu"
            - trust_remote_code (bool, optional): Trust remote code. Default: False
            - token (str, optional): Hugging Face API token

            #### Azure OpenAI models
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2
            - use_azure_openai (bool, optional): Use Azure OpenAI model. Default: False

        Raises:
            ValueError: If the model is not supported
            KeyError: If required parameters are missing
            TypeError: If the type of the parameter is incorrect
            ImportError: If the required library is not installed
        Warns:
            UserWarning: If no model_name is provided, defaulting to `text-embedding-ada-002`
        """

        # Check if the model name is provided
        if not kwds.get("model_name"):
            warnings.warn(
                f"\nNo 'model_name' specified, hence defaulting to {self.default_model}",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)

        # Check if the model is supported
        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"
        elif hf_repo_exists(
            repo_id=self.model_name,
            repo_type=kwds.get("repo_type"),
            token=kwds.get("token"),
        ):
            self.model_type = "SentenceTransformer"
        else:
            self.model_type = self.model_class.get(self.model_name)

        if not self.model_type:
            raise ValueError(
                f"Unsupported model: {self.model_name}\n"
                f"Supported models are:\n"
                f"- "
                + "\n- ".join(list(self.model_class.keys()))
                + "\n- All sentence-transformers models"
            )

        # Initialize the model
        model_handler = {
            "OpenAI": openai_embeddings.Embed,
            "AWSBedrock": awsbedrock_embeddings.Embed,
            "SentenceTransformer": stransformers_embeddings.Embed,
            "AzureOpenAI": azureopenai_embeddings.Embed,
        }
        self.model = model_handler.get(self.model_type)(**kwds)

    def __call__(
        self, user_message: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """Get embeddings for the given text(s).
        Args:
            user_message (Union[str, List[str]]): Input text or list of texts
            **kwargs: Additional keyword arguments for the embedding API
        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
        Raises:
            KeyError: If required parameters are missing
            TypeError: If the type of the parameter is incorrect
        Warns:
            UserWarning: If the model is not supported or if the model is not found in the model type
        """

        # Check if the model name is provided
        if kwds.get("model_name"):
            # Check if the model is supported in the model type class
            if self.model_class.get(kwds.get("model_name")) == self.model_type:
                model_name = kwds.get("model_name")
            elif self.model_type == "SentenceTransformer":
                warnings.warn(
                    f"\nFor using different sentence-transformers model: {kwds.get('model_name')}\n"
                    f"again reinitialize the EmbeddingModels class as currently {self.model_name} is already loaded"
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
            else:
                warnings.warn(
                    f"Unsupported model: {kwds.get('model_name')}\nSupported models are:"
                    f"\n- "
                    + "\n- ".join(list(self.model_class.keys()))
                    + "\n- All sentence-transformers models\n"
                    f"Hence defaulting to {self.model_name}",
                    UserWarning,
                )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
        else:
            model_name = self.model_name

        # Get the embeddings
        user_message_vector = self.model(
            text=user_message, model_name=model_name, **kwds
        )

        return user_message_vector


class AsyncEmbeddingModel(object):
    """Asynchronus Base class for embedding generation
    Default embedding model that will be used is `text-embedding-ada-002`"""

    default_model = DEFUALT_EMBEDDING_MODEL
    model_class = MODEL_CLASS

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Embedding Models class with the required parameters.
        Args:
            - model_name (str, optional): Name of the model to be used. Default: "gpt-4o-mini"

            ### Authentication parameters by provider:

            #### OpenAI models
            - api_key (str): OpenAI API key.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

            #### AWS Bedrock models
            - aws_access_key (str): AWS access key.
            - aws_secret_key (str): AWS secret key.
            - aws_region (str): AWS region name.
            - config (Config, optional):
                - connect_timeout (float or int, optional): The time in seconds till a timeout exception is
                thrown when attempting to make a connection. Default: 60
                - read_timeout: (float or int, optional): The time in seconds till a timeout exception is
                thrown when attempting to read from a connection. Default: 60
                - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                - retries (Dict, optional):
                    - total_max_attempts: Number of retries for the request. Default: 2

            #### Sentence Transformers models
            - model_download_path (str, optional): Path to download the model. Default: "/home/ubuntu/projects/models/embedding_models"
            - device (str, optional): Device to run the model. Default: "cpu"
            - trust_remote_code (bool, optional): Trust remote code. Default: False
            - token (str, optional): Hugging Face API token

            #### Azure OpenAI models
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2
            - use_azure_openai (bool, optional): Use Azure OpenAI model. Default: False

        Raises:
            ValueError: If the model is not supported
            KeyError: If required parameters are missing
            TypeError: If the type of the parameter is incorrect
            ImportError: If the required library is not installed
        Warns:
            UserWarning: If no model_name is provided, defaulting to `text-embedding-ada-002`
        """

        # Check if the model name is provided
        if not kwds.get("model_name"):
            warnings.warn(
                f"\nNo 'model_name' specified, hence defaulting to {self.default_model}",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)

        # Check if the model is supported
        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"
        elif hf_repo_exists(
            repo_id=self.model_name,
            repo_type=kwds.get("repo_type"),
            token=kwds.get("token"),
        ):
            self.model_type = "SentenceTransformer"
        else:
            self.model_type = self.model_class.get(self.model_name)

        if not self.model_type:
            raise ValueError(
                f"Unsupported model: {self.model_name}\n"
                f"Supported models are:\n"
                f"- "
                + "\n- ".join(list(self.model_class.keys()))
                + "\n- All sentence-transformers models"
            )

        # Initialize the model
        model_handler = {
            "OpenAI": openai_embeddings.AsyncEmbed,
            "AWSBedrock": awsbedrock_embeddings.AsyncEmbed,
            "SentenceTransformer": stransformers_embeddings.AsyncEmbed,
            "AzureOpenAI": azureopenai_embeddings.AsyncEmbed,
        }
        self.model = model_handler.get(self.model_type)(**kwds)

    async def __call__(
        self, user_message: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """Get embeddings for the given text(s).
        Args:
            user_message (Union[str, List[str]]): Input text or list of texts
            **kwargs: Additional keyword arguments for the embedding API
        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
        Raises:
            KeyError: If required parameters are missing
            TypeError: If the type of the parameter is incorrect
        Warns:
            UserWarning: If the model is not supported or if the model is not found in the model type
        """

        # Check if the model name is provided
        if kwds.get("model_name"):
            # Check if the model is supported in the model type class
            if self.model_class.get(kwds.get("model_name")) == self.model_type:
                model_name = kwds.get("model_name")
            elif self.model_type == "SentenceTransformer":
                warnings.warn(
                    f"\nFor using different sentence-transformers model: {kwds.get('model_name')}\n"
                    f"again reinitialize the EmbeddingModels class as currently {self.model_name} is already loaded"
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
            else:
                warnings.warn(
                    f"Unsupported model: {kwds.get('model_name')}\nSupported models are:"
                    f"\n- "
                    + "\n- ".join(list(self.model_class.keys()))
                    + "\n- All sentence-transformers models\n"
                    f"Hence defaulting to {self.model_name}",
                    UserWarning,
                )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
        else:
            model_name = self.model_name

        # Get the embeddings
        user_message_vector = await self.model(
            text=user_message, model_name=model_name, **kwds
        )

        return user_message_vector
