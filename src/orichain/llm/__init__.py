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
)

DEFAULT_MODEL = "gpt-4o-mini"
MODEL_CLASS = {
    "gpt-4o": "OpenAI",
    "gpt-4-turbo": "OpenAI",
    "gpt-4-turbo-preview": "OpenAI",
    "gpt-4o-mini": "OpenAI",
    "gpt-4": "OpenAI",
    "gpt-4.1": "OpenAI",
    "gpt-4.1-mini": "OpenAI",
    "gpt-4.1-nano": "OpenAI",
    "anthropic.claude-3-haiku-20240307-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-haiku-20240307-v1:0": "AnthropicAWSBedrock",
    "us-gov.anthropic.claude-3-haiku-20240307-v1:0": "AnthropicAWSBedrock",
    "eu.anthropic.claude-3-haiku-20240307-v1:0": "AnthropicAWSBedrock",
    "apac.anthropic.claude-3-haiku-20240307-v1:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-sonnet-20240229-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-sonnet-20240229-v1:0": "AnthropicAWSBedrock",
    "eu.anthropic.claude-3-sonnet-20240229-v1:0": "AnthropicAWSBedrock",
    "apac.anthropic.claude-3-sonnet-20240229-v1:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": "AnthropicAWSBedrock",
    "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0": "AnthropicAWSBedrock",
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0": "AnthropicAWSBedrock",
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "AnthropicAWSBedrock",
    "anthropic.claude-3-opus-20240229-v1:0": "AnthropicAWSBedrock",
    "us.anthropic.claude-3-opus-20240229-v1:0": "AnthropicAWSBedrock",
    "claude-3-haiku-20240307": "Anthropic",
    "claude-3-5-haiku-latest": "Anthropic",
    "claude-3-sonnet-20240229": "Anthropic",
    "claude-3-5-sonnet-latest": "Anthropic",
    "claude-3-7-sonnet-latest": "Anthropic",
    "claude-3-opus-latest": "Anthropic",
    "cohere.command-text-v14": "AWSBedrock",
    "cohere.command-light-text-v14": "AWSBedrock",
    "cohere.command-r-v1:0": "AWSBedrock",
    "cohere.command-r-plus-v1:0": "AWSBedrock",
    "meta.llama3-8b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-70b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-1-8b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-1-8b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-1-70b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-1-70b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-1-405b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-2-1b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-2-1b-instruct-v1:0": "AWSBedrock",
    "eu.meta.llama3-2-1b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-2-3b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-2-3b-instruct-v1:0": "AWSBedrock",
    "eu.meta.llama3-2-3b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-2-11b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-2-11b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-2-90b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-2-90b-instruct-v1:0": "AWSBedrock",
    "meta.llama3-3-70b-instruct-v1:0": "AWSBedrock",
    "us.meta.llama3-3-70b-instruct-v1:0": "AWSBedrock",
    "mistral.mistral-7b-instruct-v0:2": "AWSBedrock",
    "mistral.mixtral-8x7b-instruct-v0:1": "AWSBedrock",
    "mistral.mistral-large-2402-v1:0": "AWSBedrock",
    "mistral.mistral-large-2407-v1:0": "AWSBedrock",
    "mistral.mistral-small-2402-v1:0": "AWSBedrock",
    "amazon.titan-text-express-v1": "AWSBedrock",
    "amazon.titan-text-lite-v1": "AWSBedrock",
    "amazon.titan-text-premier-v1:0": "AWSBedrock",
    "amazon.nova-pro-v1:0": "AWSBedrock",
    "us.amazon.nova-pro-v1:0": "AWSBedrock",
    "amazon.nova-lite-v1:0": "AWSBedrock",
    "us.amazon.nova-lite-v1:0": "AWSBedrock",
    "amazon.nova-micro-v1:0": "AWSBedrock",
    "us.amazon.nova-micro-v1:0": "AWSBedrock",
}


class LLM(object):
    """Synchronous Language Model class for interacting with various LLM providers.

    This class provides a unified interface to interact with different language models
    from providers such as OpenAI, AWS Bedrock, Anthropic, and Azure OpenAI.
    """

    default_model = DEFAULT_MODEL
    model_class = MODEL_CLASS

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Language Model class with the required parameters.

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

            #### Anthropic models
            - api_key (str): Anthropic API key.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

            #### Azure OpenAI models
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2
            - use_azure_openai (bool, optional): Whether to use Azure OpenAI service. Default: False

        Raises:
            ValueError: If an unsupported model is specified.
            KeyError: If required parameters are not provided.
            TypeError: If an invalid type is provided for a parameter.

        Warns:
            UserWarning: If the model name is not provided, it defaults to the default model.
        """

        # Set model name, defaulting if not provided
        if not kwds.get("model_name"):
            if kwds.get("use_azure_openai"):
                warnings.warn(
                    f"\nNo 'model_name' specified, hence defaulting to {self.default_model} (Azure OpenAI)",
                    UserWarning,
                )
            else:
                warnings.warn(
                    f"\nNo 'model_name' specified, hence defaulting to {self.default_model} (OpenAI)",
                    UserWarning,
                )
        self.model_name = kwds.get("model_name", self.default_model)

        # Check if Azure OpenAI model is to be used
        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"
        else:
            # Determine model type
            self.model_type = self.model_class.get(self.model_name)

        if not self.model_type:
            raise ValueError(
                f"\nUnsupported model: {self.model_name}\nSupported models are:"
                f"\n- " + "\n- ".join(list(self.model_class.keys()))
            )

        # Initialize model handler and map model types to their respective handler classes
        model_handler = {
            "OpenAI": openai_llm.Generate,
            "AWSBedrock": awsbedrock_llm.Generate,
            "AnthropicAWSBedrock": anthropicbedrock_llm.Generate,
            "Anthropic": anthropic_llm.Generate,
            "AzureOpenAI": azureopenai_llm.Generate,
        }

        # Initialize the appropriate model handler
        self.model = model_handler.get(self.model_type)(**kwds)

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
            user_message (str): The user's input message.
            matched_sentence (List[str], optional): List of matched sentences for context.
            system_prompt (str, optional): System prompt to guide the model's behavior.
            chat_hist (List[Dict[str, str]], optional): Chat history for context.
            sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            extra_metadata (Dict, optional): Additional metadata to include in the response.
            do_json (bool, optional): Whether to return a JSON response. Default: False.
            **kwds: Additional keyword arguments to pass to the model.

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
            user_message (str): The user's input message.
            matched_sentence (List[str], optional): List of matched sentences for context.
            system_prompt (str, optional): System prompt to guide the model's behavior.
            chat_hist (List[Dict[str, str]], optional): Chat history for context.
            sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            extra_metadata (Dict, optional): Additional metadata to include in the response.
            do_json (bool, optional): Whether to return JSON responses. Default: False.
            do_sse (bool, optional): Whether to format responses as Server-Sent Events. Default: True.
            **kwds: Additional keyword arguments to pass to the model.

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

        if self.model_type == "AzureOpenAI":
            return True
        elif kwds.get("model_name"):
            if self.model_class.get(kwds.get("model_name")) == self.model_type:
                return True
            elif (
                self.model_class.get(kwds.get("model_name"))
                and self.model_class.get(kwds.get("model_name")) != self.model_type
            ):
                warnings.warn(
                    f"{kwds.get('model_name')} is a supported model but "
                    f"does not belong to {self.model_type}, again reinitialize the "
                    f"LLM class with {self.model_class.get(kwds.get('model_name'))} model class. "
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                return False
            else:
                warnings.warn(
                    f"Unsupported model: {kwds.get('model_name')}\nSupported models are:"
                    f"\n- "
                    + "\n- ".join(list(self.model_class.keys()))
                    + "\n- All sentence-transformers models\n"
                    f"Hence defaulting to {self.model_name}",
                    UserWarning,
                )
                return False
        else:
            return False


class AsyncLLM(object):
    """Asynchronous Language Model class for interacting with various LLM providers.

    This class provides a unified interface to interact with different language models
    from providers such as OpenAI, AWS Bedrock, Anthropic, and Azure OpenAI.
    """

    default_model = DEFAULT_MODEL
    model_class = MODEL_CLASS

    def __init__(self, **kwds: Any) -> None:
        """Initialize the Language Model class with the required parameters.

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

            #### Anthropic models
            - api_key (str): Anthropic API key.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

            #### Azure OpenAI models
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2
            - use_azure_openai (bool, optional): Whether to use Azure OpenAI service. Default: False

        Raises:
            ValueError: If an unsupported model is specified.
            KeyError: If required parameters are not provided.
            TypeError: If an invalid type is provided for a parameter.

        Warns:
            UserWarning: If the model name is not provided, it defaults to the default model.
        """

        # Set model name, defaulting if not provided
        if not kwds.get("model_name"):
            warnings.warn(
                f"\nNo 'model_name' specified, hence defaulting to {self.default_model} (OpenAI)",
                UserWarning,
            )
        self.model_name = kwds.get("model_name", self.default_model)

        # Determine model type
        self.model_type = self.model_class.get(self.model_name)

        # Check if Azure OpenAI model is to be used
        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"

        if not self.model_type:
            raise ValueError(
                f"\nUnsupported model: {self.model_name}\nSupported models are:"
                f"\n- " + "\n- ".join(list(self.model_class.keys()))
            )

        # Initialize model handler and map model types to their respective handler classes
        model_handler = {
            "OpenAI": openai_llm.AsyncGenerate,
            "AWSBedrock": awsbedrock_llm.AsyncGenerate,
            "AnthropicAWSBedrock": anthropicbedrock_llm.AsyncGenerate,
            "Anthropic": anthropic_llm.AsyncGenerate,
            "AzureOpenAI": azureopenai_llm.AsyncGenerate,
        }

        # Initialize the appropriate model handler
        self.model = model_handler.get(self.model_type)(**kwds)

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
            user_message (str): The user's input message.
            request (Request, optional): FastAPI request object for cancellation detection.
            matched_sentence (List[str], optional): List of matched sentences for context.
            system_prompt (str, optional): System prompt to guide the model's behavior.
            chat_hist (List[Dict[str, str]], optional): Chat history for context.
            sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            extra_metadata (Dict, optional): Additional metadata to include in the response.
            do_json (bool, optional): Whether to return a JSON response. Default: False.
            **kwds: Additional keyword arguments to pass to the model.

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
            user_message (str): The user's input message.
            request (Request, optional): FastAPI request object for cancellation detection.
            matched_sentence (List[str], optional): List of matched sentences for context.
            system_prompt (str, optional): System prompt to guide the model's behavior.
            chat_hist (List[Dict[str, str]], optional): Chat history for context.
            sampling_paras (Dict, optional): Parameters for sampling (temperature, top_p, etc.).
            extra_metadata (Dict, optional): Additional metadata to include in the response.
            do_json (bool, optional): Whether to return JSON responses. Default: False.
            do_sse (bool, optional): Whether to format responses as Server-Sent Events. Default: True.
            **kwds: Additional keyword arguments to pass to the model.

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

        if self.model_type == "AzureOpenAI":
            return True
        elif kwds.get("model_name"):
            if self.model_class.get(kwds.get("model_name")) == self.model_type:
                return True
            elif (
                self.model_class.get(kwds.get("model_name"))
                and self.model_class.get(kwds.get("model_name")) != self.model_type
            ):
                warnings.warn(
                    f"{kwds.get('model_name')} is a supported model but "
                    f"does not belong to {self.model_type}, again reinitialize the "
                    f"LLM class with {self.model_class.get(kwds.get('model_name'))} model class. "
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )
                return False
            else:
                warnings.warn(
                    f"Unsupported model: {kwds.get('model_name')}\nSupported models are:"
                    f"\n- "
                    + "\n- ".join(list(self.model_class.keys()))
                    + "\n- All sentence-transformers models\n"
                    f"Hence defaulting to {self.model_name}",
                    UserWarning,
                )
                return False
        else:
            return False
