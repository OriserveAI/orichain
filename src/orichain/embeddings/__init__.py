from typing import Any, List, Dict, Union
from orichain.embeddings import (
    openai_embeddings,
    awsbedrock_embeddings,
    stransformers_embeddings,
    azureopenai_embeddings,
    gcp_gemini_embeddings,
    gcp_vertex_embeddings,
    togetherai_embeddings,
)
import warnings
from orichain import hf_repo_exists

DEFUALT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MODEL_PROVIDER = "OpenAI"
SUPPORTED_MODELS = {
    "OpenAI": [
        "text-embedding-ada-002",
        "text-embedding-3-large",
        "text-embedding-3-small",
    ],
    "AzureOpenAI": [
        "text-embedding-ada-002",
        "text-embedding-3-large",
        "text-embedding-3-small",
    ],
    "AWSBedrock": [
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-text-v2:0",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
    ],
    "GoogleGemini": [
        "text-embedding-004",
        "gemini-embedding-exp-03-07",
        "embedding-001",
    ],
    "GoogleVertexAI": [
        "text-multilingual-embedding-002",
        "text-embedding-004",
        "text-embedding-005",
        "gemini-embedding-001",
        "gemini-embedding-exp-03-07",
        "embedding-001",
    ],
    "TogetherAI": [
        "togethercomputer/m2-bert-80M-32k-retrieval",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "Alibaba-NLP/gte-modernbert-base",
        "intfloat/multilingual-e5-large-instruct",
    ],
}


class EmbeddingModel(object):
    """Synchronus Base class for embedding generation.

    This class provides a unified interface to interact with different embedding models from providers such as OpenAI, AWS Bedrock, Google Gemini and Vertex AI, Azure OpenAI, and SentenceTransformers."""

    default_model = DEFUALT_EMBEDDING_MODEL
    default_model_provider = DEFAULT_MODEL_PROVIDER
    supported_models = SUPPORTED_MODELS
    model_handler = {
        "OpenAI": openai_embeddings.Embed,
        "AWSBedrock": awsbedrock_embeddings.Embed,
        "SentenceTransformers": stransformers_embeddings.Embed,
        "AzureOpenAI": azureopenai_embeddings.Embed,
        "GoogleGemini": gcp_gemini_embeddings.Embed,
        "GoogleVertexAI": gcp_vertex_embeddings.Embed,
        "TogetherAI": togetherai_embeddings.Embed,
    }

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Embedding Models class with the required parameters.

        Args:
            - model_name (str, optional): Name of the model to be used. Default: "text-embedding-3-small"
            - provider (str, optional): Name of the model provider. Default: "OpenAI". Allowed values:
                - OpenAI
                - AWSBedrock
                - GoogleGemini
                - GoogleVertexAI
                - AzureOpenAI
                - TogetherAI
                - SentenceTransformers

            **Authentication Arguments by provider:**

                **OpenAI models:**
                    - api_key (str): OpenAI API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **AWS Bedrock models:**
                    - aws_access_key (str): AWS access key.
                    - aws_secret_key (str): AWS secret key.
                    - aws_region (str): AWS region name.
                    - config (Config, optional):
                        - connect_timeout (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to make a connection. Default: 60
                        - read_timeout: (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to read from a connection. Default: 60
                        - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                        - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                        - retries (Dict, optional):
                            - total_max_attempts: Number of retries for the request. Default: 2

                **Google Gemini models:**
                    - api_key (str): Gemini API key
                    - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
                    - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

                **Google Vertex AI models:**
                    - api_key (str): Vertex AI API key
                    - credentials (google.auth.credentials.Credentials): The credentials to use for authentication when calling the Vertex AI APIs.
                    - project (str): The Google Cloud project ID to use for quota.
                    - location (str): The location to send API requests to (for example, us-central1).
                    - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
                    - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

                **Sentence Transformers models:**
                    - model_download_path (str, optional): Path to download the model. Default: "/home/ubuntu/projects/models/embedding_models"
                    - device (str, optional): Device to run the model. Default: "cpu"
                    - trust_remote_code (bool, optional): Trust remote code. Default: False
                    - token (str, optional): Hugging Face API token

                **Azure OpenAI models:**
                    - api_key (str): Azure OpenAI API key.
                    - azure_endpoint (str): Azure OpenAI endpoint.
                    - api_version (str): Azure OpenAI API version.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **TogetherAI models:**
                    - api_key (str): TogetherAI API key.
                    - timeout (Timeout, optional): Request timeout in seconds. Default: 60
                    - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - ValueError: If the model is not supported
            - KeyError: If required parameters are missing
            - TypeError: If the type of the parameter is incorrect
            - ImportError: If the required library is not installed

        Warns:
            - UserWarning: If no model_name is provided, defaulting to `text-embedding-3-small`
        """

        # Check if the model name is provided
        if not kwds.get("model_name"):
            warnings.warn(
                f"\nNo 'model_name' specified, hence defaulting to {self.default_model}",
                UserWarning,
            )
        if not kwds.get("provider"):
            warnings.warn(
                f"\nNo 'provider' specified, hence defaulting to {self.default_model_provider}",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)
        self.model_provider = kwds.get("provider", self.default_model_provider)

        # Validating model name and model provider name
        if self.model_provider not in self.model_handler:
            raise ValueError(
                f"\nUnsupported model provider: {self.model_provider}\nSupported providers are:"
                f"\n- " + "\n- ".join(list(self.model_handler.keys()))
            )
        elif self.model_provider == "SentenceTransformers":
            repo_check = hf_repo_exists(
                repo_id=self.model_name,
                repo_type=kwds.get("repo_type"),
                token=kwds.get("token"),
            )
            if not repo_check:
                raise ValueError(
                    f"\nThe Huggingface repository '{self.model_name}' does not exist. \nPlease ensure you provide the full repository path in 'model_name'."
                )
        elif self.model_name not in self.supported_models.get(self.model_provider):
            warnings.warn(
                f"\nModel {self.model_name} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nPlease make sure you're using the correct 'model_name' and 'provider'",
                UserWarning,
            )
        else:
            pass

        # Initialize the model
        self.model = self.model_handler.get(self.model_provider)(**kwds)

    def __call__(
        self, user_message: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """Get embeddings for the given text(s).

        Args:
            - user_message (Union[str, List[str]]): Input text or list of texts

            **Generation Arguments by provider:**

                **OpenAI & Azure OpenAI models:**
                    - model_name (str, optional): Name of the embedding model to use.

                **Google Gemini & Vertex AI models:**
                    - model_name (str, optional): Name of the embedding model to use
                    - config (google.genai.types.EmbedContentConfig, optional): Optional model configuration parameters provided to the client.models.embed_content API.

                **AWS Bedrock models:**
                    - model_name (str, optional): Name of the embedding model to use.

                    **Cohere Embedding Models:**
                        - input_type (Literal["search_query", "search_document", "classification", "clustering", "image"], optional): Type of input text. Default: "search_query"
                        - embedding_types (str, optional): Specifies the types of embeddings you want to have returned. Can be one or more of the following types: 'float', 'int8', 'uint8', 'binary', 'ubinary'
                        - truncate (Literal["NONE", "START", "END"], optional): Specifies how the API handles inputs longer than the maximum token length. Use one of the following:
                            - NONE – (Default) Returns an error when the input exceeds the maximum input token length.
                            - START – Discards the start of the input.
                            - END – Discards the end of the input.

                    **Amazon Titan Embeddings G1 Models:**
                        - dimensions (int, optional): Output dimensions. Default: 1024 (Output dimensions can be: 256, 512 and 1024)
                        - normalize (bool, optional): Normalize the output. Default: True (As recommended in docs for RAG)

                **Sentence Transformers models:**
                    - prompt_name (str, optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                      which is either set in the constructor or loaded from the model configuration. For example if
                      ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                      is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                      is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
                    - prompt (str, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                      sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                      because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
                    - output_value (Literal["sentence_embedding", "token_embeddings"], optional): The type of embeddings to return:
                      "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                      to get all output values. Defaults to "sentence_embedding".
                    - show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to False.
                    - precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                      Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                      Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                      reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
                    - batch_size (int, optional): The batch size used for the computation. Defaults to 32.
                    - convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                      Defaults to False.
                    - convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                      Defaults to False.
                    - device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
                    - normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                      the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information

        Raises:
            - KeyError: If required parameters are missing
            - TypeError: If the type of the parameter is incorrect

        Warns:
            - UserWarning: If the model is not supported or if the model is not found in the model provider
        """

        # Check if the model name is provided
        if kwds.get("model_name"):
            if self.model_provider == "SentenceTransformers":
                if kwds.get("model_name") != self.model_name:
                    warnings.warn(
                        f"\nFor using different sentence-transformers model: {kwds.get('model_name')}\n"
                        f"again reinitialize the EmbeddingModels class as currently {self.model_name} is already loaded "
                        f"Hence defaulting the model to {self.model_name}",
                        UserWarning,
                    )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
            # Check if the model is supported in the model type class
            elif kwds.get("model_name") in self.supported_models.get(
                self.model_provider
            ):
                model_name = kwds.get("model_name")
            else:
                warnings.warn(
                    f"\nModel {kwds.get('model_name')} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name'.",
                    UserWarning,
                )
                # Proceeding with the given model name regardless
                model_name = kwds.get("model_name")
            kwds.pop("model_name")
        else:
            model_name = self.model_name

        # Get the embeddings
        user_message_vector = self.model(
            text=user_message, model_name=model_name, **kwds
        )

        return user_message_vector


class AsyncEmbeddingModel(object):
    """Asynchronus Base class for embedding generation.

    This class provides a unified interface to interact with different embedding models from providers such as OpenAI, AWS Bedrock, Google Gemini and Vertex AI, Azure OpenAI, and SentenceTransformers."""

    default_model = DEFUALT_EMBEDDING_MODEL
    default_model_provider = DEFAULT_MODEL_PROVIDER
    supported_models = SUPPORTED_MODELS
    model_handler = {
        "OpenAI": openai_embeddings.AsyncEmbed,
        "AWSBedrock": awsbedrock_embeddings.AsyncEmbed,
        "SentenceTransformers": stransformers_embeddings.AsyncEmbed,
        "AzureOpenAI": azureopenai_embeddings.AsyncEmbed,
        "GoogleGemini": gcp_gemini_embeddings.AsyncEmbed,
        "GoogleVertexAI": gcp_vertex_embeddings.AsyncEmbed,
        "TogetherAI": togetherai_embeddings.AsyncEmbed,
    }

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Embedding Models class with the required parameters.

        Args:
            - model_name (str, optional): Name of the model to be used. Default: "text-embedding-3-small"
            - provider (str, optional): Name of the model provider. Default: "OpenAI". Allowed values:
                - OpenAI
                - AWSBedrock
                - GoogleGemini
                - GoogleVertexAI
                - AzureOpenAI
                - TogetherAI
                - SentenceTransformers

            **Authentication Arguments by provider:**

                **OpenAI models:**
                    - api_key (str): OpenAI API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **AWS Bedrock models:**
                    - aws_access_key (str): AWS access key.
                    - aws_secret_key (str): AWS secret key.
                    - aws_region (str): AWS region name.
                    - config (Config, optional):
                        - connect_timeout (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to make a connection. Default: 60
                        - read_timeout: (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to read from a connection. Default: 60
                        - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                        - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                        - retries (Dict, optional):
                            - total_max_attempts: Number of retries for the request. Default: 2

                **Google Gemini models:**
                    - api_key (str): Gemini API key
                    - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
                    - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

                **Google Vertex AI models:**
                    - api_key (str): Vertex AI API key
                    - credentials (google.auth.credentials.Credentials): The credentials to use for authentication when calling the Vertex AI APIs.
                    - project (str): The Google Cloud project ID to use for quota.
                    - location (str): The location to send API requests to (for example, us-central1).
                    - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
                    - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

                **Sentence Transformers models:**
                    - model_download_path (str, optional): Path to download the model. Default: "/home/ubuntu/projects/models/embedding_models"
                    - device (str, optional): Device to run the model. Default: "cpu"
                    - trust_remote_code (bool, optional): Trust remote code. Default: False
                    - token (str, optional): Hugging Face API token

                **Azure OpenAI models:**
                    - api_key (str): Azure OpenAI API key.
                    - azure_endpoint (str): Azure OpenAI endpoint.
                    - api_version (str): Azure OpenAI API version.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **TogetherAI models:**
                    - api_key (str): TogetherAI API key.
                    - timeout (Timeout, optional): Request timeout in seconds. Default: 60
                    - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - ValueError: If the model is not supported
            - KeyError: If required parameters are missing
            - TypeError: If the type of the parameter is incorrect
            - ImportError: If the required library is not installed

        Warns:
            - UserWarning: If no model_name is provided, defaulting to `text-embedding-3-small`
        """

        # Check if the model name is provided
        if not kwds.get("model_name"):
            warnings.warn(
                f"\nNo 'model_name' specified, hence defaulting to {self.default_model}",
                UserWarning,
            )
        if not kwds.get("provider"):
            warnings.warn(
                f"\nNo 'provider' specified, hence defaulting to {self.default_model_provider}",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)
        self.model_provider = kwds.get("provider", self.default_model_provider)

        # Validating model name and model provider name
        if self.model_provider not in self.model_handler:
            raise ValueError(
                f"\nUnsupported model provider: {self.model_provider}\nSupported providers are:"
                f"\n- " + "\n- ".join(list(self.model_handler.keys()))
            )
        elif self.model_provider == "SentenceTransformers":
            repo_check = hf_repo_exists(
                repo_id=self.model_name,
                repo_type=kwds.get("repo_type"),
                token=kwds.get("token"),
            )
            if not repo_check:
                raise ValueError(
                    f"\nThe Huggingface repository '{self.model_name}' does not exist. \nPlease ensure you provide the full repository path in 'model_name'."
                )
        elif self.model_name not in self.supported_models.get(self.model_provider):
            warnings.warn(
                f"\nModel {self.model_name} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nPlease make sure you're using the correct 'model_name' and 'provider'",
                UserWarning,
            )
        else:
            pass

        # Initialize the model
        self.model = self.model_handler.get(self.model_provider)(**kwds)

    async def __call__(
        self, user_message: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """Get embeddings for the given text(s).

        Args:
            - user_message (Union[str, List[str]]): Input text or list of texts

            **Generation Arguments by provider:**

                **OpenAI & Azure OpenAI models:**
                    - model_name (str, optional): Name of the embedding model to use.

                **Google Gemini & Vertex AI models:**
                    - model_name (str, optional): Name of the embedding model to use
                    - config (google.genai.types.EmbedContentConfig, optional): Optional model configuration parameters provided to the client.models.embed_content API.

                **AWS Bedrock models:**
                    - model_name (str, optional): Name of the embedding model to use.

                    **Cohere Embedding Models:**
                        - input_type (Literal["search_query", "search_document", "classification", "clustering", "image"], optional): Type of input text. Default: "search_query"
                        - embedding_types (str, optional): Specifies the types of embeddings you want to have returned. Can be one or more of the following types: 'float', 'int8', 'uint8', 'binary', 'ubinary'
                        - truncate (Literal["NONE", "START", "END"], optional): Specifies how the API handles inputs longer than the maximum token length. Use one of the following:
                            - NONE – (Default) Returns an error when the input exceeds the maximum input token length.
                            - START – Discards the start of the input.
                            - END – Discards the end of the input.

                    **Amazon Titan Embeddings G1 Models:**
                        - dimensions (int, optional): Output dimensions. Default: 1024 (Output dimensions can be: 256, 512 and 1024)
                        - normalize (bool, optional): Normalize the output. Default: True (As recommended in docs for RAG)

                **Sentence Transformers models:**
                    - prompt_name (str, optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                      which is either set in the constructor or loaded from the model configuration. For example if
                      ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                      is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                      is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
                    - prompt (str, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                      sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                      because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
                    - output_value (Literal["sentence_embedding", "token_embeddings"], optional): The type of embeddings to return:
                      "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                      to get all output values. Defaults to "sentence_embedding".
                    - show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to False.
                    - precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                      Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                      Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                      reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
                    - batch_size (int, optional): The batch size used for the computation. Defaults to 32.
                    - convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                      Defaults to False.
                    - convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                      Defaults to False.
                    - device (str, optional): Which :class:`torch.device` to use for the computation. Defaults to None.
                    - normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                      the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.

        Returns:
            (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information

        Raises:
            - KeyError: If required parameters are missing
            - TypeError: If the type of the parameter is incorrect

        Warns:
            - UserWarning: If the model is not supported or if the model is not found in the model provider
        """

        # Check if the model name is provided
        if kwds.get("model_name"):
            if self.model_provider == "SentenceTransformers":
                if kwds.get("model_name") != self.model_name:
                    warnings.warn(
                        f"\nFor using different sentence-transformers model: {kwds.get('model_name')}\n"
                        f"again reinitialize the EmbeddingModels class as currently {self.model_name} is already loaded "
                        f"Hence defaulting the model to {self.model_name}",
                        UserWarning,
                    )
                # Defaulting to the model_name that is already loaded
                model_name = self.model_name
            # Check if the model is supported in the model type class
            elif kwds.get("model_name") in self.supported_models.get(
                self.model_provider
            ):
                model_name = kwds.get("model_name")
            else:
                warnings.warn(
                    f"\nModel {kwds.get('model_name')} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name'.",
                    UserWarning,
                )
                # Proceeding with the given model name regardless
                model_name = kwds.get("model_name")
            kwds.pop("model_name")
        else:
            model_name = self.model_name

        # Get the embeddings
        user_message_vector = await self.model(
            text=user_message, model_name=model_name, **kwds
        )

        return user_message_vector
