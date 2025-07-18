from typing import Any, Optional, List, Dict, Generator, AsyncGenerator
import warnings
import json
from fastapi import Request

from orichain import error_explainer

from orichain.llm import (
    openai_llm,
    anthropicbedrock_llm,
    anthropic_llm,
    awsbedrock_llm,
    azureopenai_llm,
    gcp_gemini_llm,
    gcp_vertex_llm,
)

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_MODEL_PROVIDER = "OpenAI"
SUPPORTED_MODELS = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "AzureOpenAI": [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "AnthropicBedrock": [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "us.anthropic.claude-3-haiku-20240307-v1:0",
        "us-gov.anthropic.claude-3-haiku-20240307-v1:0",
        "eu.anthropic.claude-3-haiku-20240307-v1:0",
        "apac.anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "eu.anthropic.claude-3-sonnet-20240229-v1:0",
        "apac.anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "us.anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",
        "us.anthropic.claude-opus-4-20250514-v1:0",
    ],
    "Anthropic": [
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-latest",
        "claude-3-sonnet-20240229",
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-20250514",
        "claude-3-opus-latest",
        "claude-opus-4-20250514",
    ],
    "AWSBedrock": [
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.command-r-v1:0",
        "cohere.command-r-plus-v1:0",
        "meta.llama3-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-1-8b-instruct-v1:0",
        "us.meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "us.meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-1-405b-instruct-v1:0",
        "meta.llama3-2-1b-instruct-v1:0",
        "us.meta.llama3-2-1b-instruct-v1:0",
        "eu.meta.llama3-2-1b-instruct-v1:0",
        "meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "eu.meta.llama3-2-3b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-2-90b-instruct-v1:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-3-70b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        "meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "meta.llama4-scout-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-large-2407-v1:0",
        "mistral.mistral-small-2402-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-text-premier-v1:0",
        "amazon.nova-pro-v1:0",
        "us.amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "us.amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
        "us.amazon.nova-micro-v1:0",
    ],
    "GoogleGemini": [
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "GoogleVertexAI": [
        "gemini-1.5-pro",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
}


class LLM(object):
    """Synchronous Language Model class for interacting with various LLM providers.

    This class provides a unified interface to interact with different language models
    from providers such as OpenAI, AWS Bedrock, Google Gemini and Vertex AI, Anthropic, and Azure OpenAI.
    """

    default_model = DEFAULT_MODEL
    default_model_provider = DEFAULT_MODEL_PROVIDER
    supported_models = SUPPORTED_MODELS
    model_handler = {
        "OpenAI": openai_llm.Generate,
        "AWSBedrock": awsbedrock_llm.Generate,
        "AnthropicBedrock": anthropicbedrock_llm.Generate,
        "Anthropic": anthropic_llm.Generate,
        "AzureOpenAI": azureopenai_llm.Generate,
        "GoogleGemini": gcp_gemini_llm.Generate,
        "GoogleVertexAI": gcp_vertex_llm.Generate,
    }

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Language Model class with the required parameters.

        Args:
            - model_name (str, optional): Name of the model to be used. Default: "gpt-4.1-mini"
            - provider (str, optional): Name of the model provider. Default: "OpenAI". Allowed values:
                - OpenAI
                - AzureOpenAI
                - AWSBedrock
                - GoogleGemini
                - GoogleVertexAI
                - AnthropicBedrock
                - Anthropic

            **Authentication Arguments by provider:**

                **OpenAI models:**
                    - api_key (str): OpenAI API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **AWS Bedrock models:**
                    - aws_access_key (str): AWS access key.
                    - aws_secret_key (str): AWS secret key.
                    - aws_region (str): AWS region name.
                    - prompt_caching (bool, optional): Whether to use prompt caching. Default: True
                    - config (botocore.config.Config, optional):
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

                **Anthropic models:**
                    - api_key (str): Anthropic API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2
                    - prompt_caching (bool, optional): Whether to use prompt caching. Default: True

                **Azure OpenAI models:**
                    - api_key (str): Azure OpenAI API key.
                    - azure_endpoint (str): Azure OpenAI endpoint.
                    - api_version (str): Azure OpenAI API version.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - ValueError: If an unsupported model is specified.
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter.

        Warns:
            - UserWarning: If the model name is not provided, it defaults to the default model.
        """

        # Set model name and model provider, defaulting if not provided
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
        elif self.model_name not in self.supported_models.get(self.model_provider):
            warnings.warn(
                f"\nModel {self.model_name} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name' and 'provider'",
                UserWarning,
            )

        # Initialize the appropriate model handler
        self.model = self.model_handler.get(self.model_provider)(**kwds)

    def __call__(
        self,
        user_message: str,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
        sampling_paras: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None,
        do_json: bool = False,
        **kwds: Any,
    ) -> Dict:
        """Generate a synchronous response from the language model.

        Args:
            - user_message (str): The user's input message.
            - system_prompt (str, optional): System prompt to guide the model's behavior.
            - chat_hist (List[Dict[str, str]], optional): Chat history for context.
            - sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            - do_json (bool, optional): Whether to return a JSON response. Default: False.
            - model_name (str, optional): Specifies the model to use. If not provided, the default is the model set during class instantiation.
            - matched_sentence (List[str], optional): A list of matched text chunks for context. Not used internally, but included in the response under the matched_sentence key.
            - extra_metadata (Dict, optional): Additional metadata to include in the response.

            **Generation Arguments by provider:**

                **AWS Bedrock models:**
                    - additional_model_fields (Dict, optional): additionalModelRequestFields passed to the client in the request body.

                **Google Gemini & Vertex AI models:**
                    - config (google.genai.types.GenerateContentConfig, optional): Optional model configuration parameters provided to the client.chats.create API.
                    - response_mime_type (str, optional): Output response mimetype of the generated candidate text. Supported mimetype: "text/plain" (Default), "application/json" (if do_json=True)

                **Anthropic & AnthropicBedrock models:**
                    - timeout (httpx.Timeout, optional): - timeout (httpx.Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0

        Returns:
            Dict: The model's response with metadata.
        """
        try:
            # Handle model switching if a different model is specified in kwds
            if self._model_n_model_type_validator(**kwds):
                model_name = kwds.pop("model_name", self.model_name)
            else:
                model_name = self.model_name

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}
            extra_metadata = extra_metadata or {}

            # Generate the response
            result = self.model(
                model_name=model_name,
                user_message=user_message,
                system_prompt=system_prompt,
                chat_hist=chat_hist,
                sampling_paras=sampling_paras,
                do_json=do_json,
                **kwds,
            )

            # Add user message and matched sentence to the response
            if "error" not in result:
                result.update({"message": user_message})
                if matched_sentence:
                    result.update({"matched_sentence": matched_sentence})
                # Add extra metadata to the response
                if extra_metadata:
                    result["metadata"].update(extra_metadata)

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def stream(
        self,
        user_message: str,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: List = None,
        sampling_paras: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None,
        do_json: bool = False,
        do_sse: bool = True,
        **kwds: Any,
    ) -> Generator:
        """Stream responses from the language model.

        Args:
            - user_message (str): The user's input message.
            - system_prompt (str, optional): System prompt to guide the model's behavior.
            - chat_hist (List[Dict[str, str]], optional): Chat history for context.
            - sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            - do_json (bool, optional): Whether to return JSON responses. Default: False.
            - do_sse (bool, optional): Whether to format responses as Server-Sent Events. Default: True.
            - model_name (str, optional): Specifies the model to use. If not provided, the default is the model set during class instantiation.
            - matched_sentence (List[str], optional): A list of matched text chunks for context. Not used internally, but included in the response under the matched_sentence key.
            - extra_metadata (Dict, optional): Additional metadata to include in the response.

            **Generation Arguments by provider:**

                **AWS Bedrock models:**
                    - additional_model_fields (Dict, optional): additionalModelRequestFields passed to the client in the request body.

                **Google Gemini & Vertex AI models:**
                    - config (google.genai.types.GenerateContentConfig, optional): Optional model configuration parameters provided to the client.chats.create API.
                    - response_mime_type (str, optional): Output response mimetype of the generated candidate text. Supported mimetype: "text/plain" (Default), "application/json" (if do_json=True)

        Yields:
            Generator: Stream of responses from the language model.
        """
        try:
            # Handle model switching if a different model is specified in kwds
            if self._model_n_model_type_validator(**kwds):
                model_name = kwds.get("model_name", self.model_name)
            else:
                model_name = self.model_name

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}
            extra_metadata = extra_metadata or {}

            # Stream responses from the model
            result = self.model.streaming(
                model_name=model_name,
                user_message=user_message,
                system_prompt=system_prompt,
                chat_hist=chat_hist,
                sampling_paras=sampling_paras,
                do_json=do_json,
                **kwds,
            )

            # Process each chunk in the stream
            for chunk in result:
                if isinstance(chunk, str):
                    if do_sse:
                        yield self._format_sse(chunk, event="text")
                    else:
                        yield chunk
                elif isinstance(chunk, Dict):
                    if "error" not in chunk:
                        chunk.update(
                            {
                                "message": user_message,
                            }
                        )
                        if matched_sentence:
                            chunk.update({"matched_sentence": matched_sentence})
                        if extra_metadata:
                            chunk["metadata"].update(extra_metadata)
                    if do_sse:
                        yield self._format_sse(chunk, event="body")
                    else:
                        yield chunk

        except Exception as e:
            error_explainer(e)
            yield self._format_sse({"error": 500, "reason": str(e)}, event="body")

    def _format_sse(self, data: Any, event=None) -> str:
        """Format data for Server-Sent Events (SSE).

        Args:
            data (Any): The data to format.
            event (str, optional): The event type.

        Returns:
            str: Formatted SSE message.
        """
        msg = f"data: {json.dumps(data)}\n\n"

        if event is not None:
            msg = f"event: {event}\n{msg}"

        return msg

    def _model_n_model_type_validator(self, **kwds: Any) -> bool:
        """Validate if the requested model is compatible with the current model type.

        Args:
            **kwds: Keyword arguments that may contain a 'model_name'.

        Returns:
            bool: True if the model is compatible, False otherwise.
        """

        if kwds.get("model_name"):
            if kwds.get("model_name") in self.supported_models.get(self.model_provider):
                return True
            elif kwds.get("model_name") in [
                item for sublist in self.supported_models.values() for item in sublist
            ]:
                warnings.warn(
                    f"{kwds.get('model_name')} is a supported model but does not belong to {self.model_provider} provider. "
                    f"Please reinitialize the LLM class with the '{kwds.get('model_name')}' model and the correct provider. "
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                return False
            else:
                warnings.warn(
                    f"\nModel {kwds.get('model_name')} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name'",
                    UserWarning,
                )
                return True
        else:
            return False


class AsyncLLM(object):
    """Asynchronous Language Model class for interacting with various LLM providers.

    This class provides a unified interface to interact with different language models
    from providers such as OpenAI, AWS Bedrock, Google Gemini and Vertex AI, Anthropic, and Azure OpenAI.
    """

    default_model = DEFAULT_MODEL
    default_model_provider = DEFAULT_MODEL_PROVIDER
    supported_models = SUPPORTED_MODELS
    model_handler = model_handler = {
        "OpenAI": openai_llm.AsyncGenerate,
        "AWSBedrock": awsbedrock_llm.AsyncGenerate,
        "AnthropicBedrock": anthropicbedrock_llm.AsyncGenerate,
        "Anthropic": anthropic_llm.AsyncGenerate,
        "AzureOpenAI": azureopenai_llm.AsyncGenerate,
        "GoogleGemini": gcp_gemini_llm.AsyncGenerate,
        "GoogleVertexAI": gcp_vertex_llm.AsyncGenerate,
    }

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Language Model class with the required parameters.

        Args:
            - model_name (str, optional): Name of the model to be used. Default: "gpt-4.1-mini"
            - provider (str, optional): Name of the model provider. Default: "OpenAI". Allowed values:
                - OpenAI
                - AzureOpenAI
                - AWSBedrock
                - GoogleGemini
                - GoogleVertexAI
                - AnthropicBedrock
                - Anthropic

            **Authentication Arguments by provider:**

                **OpenAI models:**
                    - api_key (str): OpenAI API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

                **AWS Bedrock models:**
                    - aws_access_key (str): AWS access key.
                    - aws_secret_key (str): AWS secret key.
                    - aws_region (str): AWS region name.
                    - prompt_caching (bool, optional): Whether to use prompt caching. Default: True
                    - config (botocore.config.Config, optional):
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

                **Anthropic models:**
                    - api_key (str): Anthropic API key.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2
                    - prompt_caching (bool, optional): Whether to use prompt caching. Default: True

                **Azure OpenAI models:**
                    - api_key (str): Azure OpenAI API key.
                    - azure_endpoint (str): Azure OpenAI endpoint.
                    - api_version (str): Azure OpenAI API version.
                    - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
                    - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - ValueError: If an unsupported model is specified.
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter.

        Warns:
            - UserWarning: If the model name is not provided, it defaults to the default model.
        """

        # Set model name and model provider, defaulting if not provided
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
        elif self.model_name not in self.supported_models.get(self.model_provider):
            warnings.warn(
                f"\nModel {self.model_name} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name' and 'provider'",
                UserWarning,
            )

        # Initialize the appropriate model handler
        self.model = self.model_handler.get(self.model_provider)(**kwds)

    async def __call__(
        self,
        user_message: str,
        request: Optional[Request] = None,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
        sampling_paras: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None,
        do_json: bool = False,
        **kwds: Any,
    ) -> Dict:
        """Generate a synchronous response from the language model.

        Args:
            - user_message (str): The user's input message.
            - system_prompt (str, optional): System prompt to guide the model's behavior.
            - chat_hist (List[Dict[str, str]], optional): Chat history for context.
            - sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            - do_json (bool, optional): Whether to return a JSON response. Default: False.
            - model_name (str, optional): Specifies the model to use. If not provided, the default is the model set during class instantiation.
            - request (Request, optional): FastAPI Request object for cancellation detection.
            - matched_sentence (List[str], optional): A list of matched text chunks for context. Not used internally, but included in the response under the matched_sentence key.
            - extra_metadata (Dict, optional): Additional metadata to include in the response.

            **Generation Arguments by provider:**

                **AWS Bedrock models:**
                    - additional_model_fields (Dict, optional): additionalModelRequestFields passed to the client in the request body.

                **Google Gemini & Vertex AI models:**
                    - config (google.genai.types.GenerateContentConfig, optional): Optional model configuration parameters provided to the client.chats.create API.
                    - response_mime_type (str, optional): Output response mimetype of the generated candidate text. Supported mimetype: "text/plain" (Default), "application/json" (if do_json=True)

                **Anthropic & AnthropicBedrock models:**
                    - timeout (httpx.Timeout, optional): - timeout (httpx.Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0

        Returns:
            Dict: The model's response with metadata.
        """
        try:
            # Handle model switching if a different model is specified in kwds
            if await self._model_n_model_type_validator(**kwds):
                model_name = kwds.pop("model_name", self.model_name)
            else:
                model_name = self.model_name

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}
            extra_metadata = extra_metadata or {}

            # Check if request is disconnected
            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            # Generate the response
            result = await self.model(
                request=request,
                model_name=model_name,
                user_message=user_message,
                system_prompt=system_prompt,
                chat_hist=chat_hist,
                sampling_paras=sampling_paras,
                do_json=do_json,
                **kwds,
            )

            # Add user message and matched sentence to the response
            if "error" not in result:
                result.update({"message": user_message})
                if matched_sentence:
                    result.update({"matched_sentence": matched_sentence})
                # Add extra metadata to the response
                if extra_metadata:
                    result["metadata"].update(extra_metadata)

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def stream(
        self,
        user_message: str,
        request: Optional[Request] = None,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: List = None,
        sampling_paras: Optional[Dict] = None,
        extra_metadata: Optional[Dict] = None,
        do_json: bool = False,
        do_sse: bool = True,
        **kwds: Any,
    ) -> AsyncGenerator:
        """Stream responses from the language model.

        Args:
            - user_message (str): The user's input message.
            - system_prompt (str, optional): System prompt to guide the model's behavior.
            - chat_hist (List[Dict[str, str]], optional): Chat history for context.
            - sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            - do_json (bool, optional): Whether to return JSON responses. Default: False.
            - do_sse (bool, optional): Whether to format responses as Server-Sent Events. Default: True.
            - model_name (str, optional): Specifies the model to use. If not provided, the default is the model set during class instantiation.
            - request (Request, optional): FastAPI Request object for cancellation detection.
            - matched_sentence (List[str], optional): A list of matched text chunks for context. Not used internally, but included in the response under the matched_sentence key.
            - extra_metadata (Dict, optional): Additional metadata to include in the response.

            **Generation Arguments by provider:**

                **AWS Bedrock models:**
                    - additional_model_fields (Dict, optional): additionalModelRequestFields passed to the client in the request body.

                **Google Gemini & Vertex AI models:**
                    - config (google.genai.types.GenerateContentConfig, optional): Optional model configuration parameters provided to the client.chats.create API.
                    - response_mime_type (str, optional): Output response mimetype of the generated candidate text. Supported mimetype: "text/plain" (Default), "application/json" (if do_json=True)

        Yields:
            AsyncGenerator: Stream of responses from the language model.
        """
        try:
            # Handle model switching if a different model is specified in kwds
            if await self._model_n_model_type_validator(**kwds):
                model_name = kwds.get("model_name", self.model_name)
            else:
                model_name = self.model_name

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}
            extra_metadata = extra_metadata or {}

            # Check if the request has been disconnected
            if request and await request.is_disconnected():
                yield await self._format_sse(
                    {"error": 400, "reason": "request aborted by user"}, event="body"
                )
            else:
                # Stream responses from the model
                result = self.model.streaming(
                    request=request,
                    model_name=model_name,
                    user_message=user_message,
                    system_prompt=system_prompt,
                    chat_hist=chat_hist,
                    sampling_paras=sampling_paras,
                    do_json=do_json,
                    **kwds,
                )

                # Process each chunk in the stream
                async for chunk in result:
                    if isinstance(chunk, str):
                        if do_sse:
                            yield await self._format_sse(chunk, event="text")
                        else:
                            yield chunk
                    elif isinstance(chunk, Dict):
                        if "error" not in chunk:
                            chunk.update(
                                {
                                    "message": user_message,
                                }
                            )
                            if matched_sentence:
                                chunk.update({"matched_sentence": matched_sentence})
                            if extra_metadata:
                                chunk["metadata"].update(extra_metadata)
                        if do_sse:
                            yield await self._format_sse(chunk, event="body")
                        else:
                            yield chunk

        except Exception as e:
            error_explainer(e)
            yield await self._format_sse({"error": 500, "reason": str(e)}, event="body")

    async def _format_sse(self, data: Any, event=None) -> str:
        """Format data for Server-Sent Events (SSE).

        Args:
            data (Any): The data to format.
            event (str, optional): The event type.

        Returns:
            str: Formatted SSE message.
        """
        msg = f"data: {json.dumps(data)}\n\n"

        if event is not None:
            msg = f"event: {event}\n{msg}"

        return msg

    async def _model_n_model_type_validator(self, **kwds: Any) -> bool:
        """Validate if the requested model is compatible with the current model type.

        Args:
            **kwds: Keyword arguments that may contain a 'model_name'.

        Returns:
            bool: True if the model is compatible, False otherwise.
        """

        if kwds.get("model_name"):
            if kwds.get("model_name") in self.supported_models.get(self.model_provider):
                return True
            elif kwds.get("model_name") in [
                item for sublist in self.supported_models.values() for item in sublist
            ]:
                warnings.warn(
                    f"{kwds.get('model_name')} is a supported model but does not belong to {self.model_provider} provider. "
                    f"Please reinitialize the AsyncLLM class with the '{kwds.get('model_name')}' model and the correct provider. "
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                return False
            else:
                warnings.warn(
                    f"\nModel {kwds.get('model_name')} for provider {self.model_provider} is not supported by Orichain. Supported models for {self.model_provider} are: [{', '.join(self.supported_models.get(self.model_provider))}] \nUsing an unsupported model may lead to unexpected issues. Please verify that you are using the correct 'model_name'",
                    UserWarning,
                )
                return True
        else:
            return False
