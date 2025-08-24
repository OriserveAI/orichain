import json
from typing import Dict, List, Optional, Generator, AsyncGenerator
from fastapi import Request

from orichain import error_explainer


class Generate(object):
    """
    Synchronous wrapper for Azure OpenAI's API client.

    This class provides methods for generating responses from OpenAI's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Azure OpenAI client and set up API key.

        Args:
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from httpx import Timeout

        # Validate input parameters
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

        # Initialize the Azure OpenAI client with provided parameters
        from openai import AzureOpenAI
        import tiktoken

        self.client = AzureOpenAI(
            azure_endpoint=kwds.get("azure_endpoint"),
            api_key=kwds.get("api_key"),
            api_version=kwds.get("api_version"),
            timeout=kwds.get("timeout")
            or Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            max_retries=kwds.get("max_retries") or 2,
        )
        self.tiktoken = tiktoken

    def __call__(
        self,
        model_name: str,
        user_message: str,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the Azure OpenAI model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Check if tools and tool_choice are provided and format them
            tools = (
                [{"type": "function", "function": tool} for tool in tools]
                if tools
                else []
            )
            if tool_choice and tool_choice not in ["none", "auto", "required"]:
                if tool_choice in [
                    tool.get("function", {}).get("name") for tool in tools
                ]:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }

            # Call the OpenAI API with the formatted messages
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format={"type": "json_object"}
                if do_json
                else {"type": "text"},
                **sampling_paras,
            )

            result = {
                "response": completion.choices[0].message.content or "",
                "metadata": {"usage": completion.usage.to_dict()},
            }

            if tools:
                tool_calls = (
                    [
                        tool_call.to_dict()
                        for tool_call in completion.choices[0].message.tool_calls
                    ]
                    if completion.choices[0].message.tool_calls
                    else []
                )
                for tool_call in tool_calls:
                    tool_call["function"]["arguments"] = (
                        json.loads(tool_call["function"]["arguments"])
                        if tool_call["function"]["arguments"]
                        else {}
                    )
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def streaming(
        self,
        model_name: str,
        user_message: str,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> Generator:
        """
        Stream responses from the specified model.

        Args:
            - model_name (str): Name of the Azure OpenAI model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False

        Yields:
            Generator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages

            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Check if tools and tool_choice are provided and format them
                tools = (
                    [{"type": "function", "function": tool} for tool in tools]
                    if tools
                    else []
                )
                if tool_choice and tool_choice not in ["none", "auto", "required"]:
                    if tool_choice in [
                        tool.get("function", {}).get("name") for tool in tools
                    ]:
                        tool_choice = {
                            "type": "function",
                            "function": {"name": tool_choice},
                        }
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools."
                        )

                # Start the streaming session
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format={"type": "json_object"}
                    if do_json
                    else {"type": "text"},
                    stream_options={"include_usage": True},
                    **sampling_paras,
                )

                response = ""
                usage = {}
                tool_calls = []
                tool_arg_buffer = ""

                # Stream text chunks as they become available
                for chunk in completion:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        response += delta.content
                        yield delta.content
                    elif delta and delta.tool_calls:
                        tc = delta.tool_calls[0]
                        if tc.id:
                            tool_calls.append(tc.to_dict())
                        if tc.function and tc.function.arguments:
                            tool_arg_buffer += tc.function.arguments
                            try:
                                tool_calls[-1]["function"]["arguments"] = json.loads(
                                    tool_arg_buffer
                                )
                                tool_arg_buffer = ""
                            except json.JSONDecodeError:
                                continue
                    elif chunk.usage:
                        usage = chunk.usage.to_dict()

                # Format the final response with metadata
                result = {
                    "response": response,
                    "metadata": {"usage": usage},
                }

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    def _chat_formatter(
        self,
        user_message: str,
        chat_hist: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """
        Format user messages and chat history for the Azure OpenAI API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by Azure OpenAI's API

        Raises:
            KeyError: If the user message format is invalid
        """
        try:
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add chat history if provided
            if chat_hist:
                messages.extend(chat_hist)

            # Add user message
            messages.append({"role": "user", "content": user_message})

            return messages

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def num_tokens_from_string(
        self, model_name: str = "gpt-3.5-turbo", string: str = None
    ) -> int:
        """Returns the number of tokens in a text string.

        Args:
        model_name (str): The tiktoken tokenizer for specifed model name
        string (str): String to calculate the tokens for

        Returns:
        int: Number of tokens"""
        if string and model_name:
            encoding = self.tiktoken.encoding_for_model(model_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        else:
            return 0


class AsyncGenerate(object):
    """
    Asynchronous wrapper for Azure OpenAI's API client.

    This class provides methods for generating responses from Azure OpenAI's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Azure OpenAI client and set up API key.

        Args:
            - api_key (str): Azure OpenAI API key.
            - azure_endpoint (str): Azure OpenAI endpoint.
            - api_version (str): Azure OpenAI API version.
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default: 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default: 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from httpx import Timeout

        # Validate input parameters
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

        # Initialize the Azure OpenAI client with provided parameters
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
        self,
        model_name: str,
        user_message: str,
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the OpenAI model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Check if the request was disconnected
            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            # Check if tools and tool_choice are provided and format them
            tools = (
                [{"type": "function", "function": tool} for tool in tools]
                if tools
                else []
            )
            if tool_choice and tool_choice not in ["none", "auto", "required"]:
                if tool_choice in [
                    tool.get("function", {}).get("name") for tool in tools
                ]:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }

            # Call the Azure OpenAI API with the formatted messages
            completion = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format={"type": "json_object"}
                if do_json
                else {"type": "text"},
                **sampling_paras,
            )

            result = {
                "response": completion.choices[0].message.content or "",
                "metadata": {"usage": completion.usage.to_dict()},
            }

            if tools:
                tool_calls = (
                    [
                        tool_call.to_dict()
                        for tool_call in completion.choices[0].message.tool_calls
                    ]
                    if completion.choices[0].message.tool_calls
                    else []
                )
                for tool_call in tool_calls:
                    tool_call["function"]["arguments"] = (
                        json.loads(tool_call["function"]["arguments"])
                        if tool_call["function"]["arguments"]
                        else {}
                    )
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def streaming(
        self,
        model_name: str,
        user_message: str,
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> AsyncGenerator:
        """
        Stream responses from the specified model.

        Args:
            - model_name (str): Name of the OpenAI model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False

        Yields:
            AsyncGenerator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages

            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Check if tools and tool_choice are provided and format them
                tools = (
                    [{"type": "function", "function": tool} for tool in tools]
                    if tools
                    else []
                )
                if tool_choice and tool_choice not in ["none", "auto", "required"]:
                    if tool_choice in [
                        tool.get("function", {}).get("name") for tool in tools
                    ]:
                        tool_choice = {
                            "type": "function",
                            "function": {"name": tool_choice},
                        }
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools."
                        )

                # Start the streaming session
                completion = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format={"type": "json_object"}
                    if do_json
                    else {"type": "text"},
                    stream_options={"include_usage": True},
                    **sampling_paras,
                )

                response = ""
                usage = {}
                tool_calls = []
                tool_arg_buffer = ""

                # Stream text chunks as they become available
                async for chunk in completion:
                    if request and await request.is_disconnected():
                        yield {"error": 400, "reason": "request aborted by user"}
                        await completion.close()
                        break
                    else:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and delta.content:
                            response += delta.content
                            yield delta.content
                        elif delta and delta.tool_calls:
                            tc = delta.tool_calls[0]
                            if tc.id:
                                tool_calls.append(tc.to_dict())
                            if tc.function and tc.function.arguments:
                                tool_arg_buffer += tc.function.arguments
                                try:
                                    tool_calls[-1]["function"]["arguments"] = (
                                        json.loads(tool_arg_buffer)
                                    )
                                    tool_arg_buffer = ""
                                except json.JSONDecodeError:
                                    continue
                        elif chunk.usage:
                            usage = chunk.usage.to_dict()

                # Format the final response with metadata
                result = {
                    "response": response,
                    "metadata": {"usage": usage},
                }

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    async def _chat_formatter(
        self,
        user_message: str,
        chat_hist: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """
        Format user messages and chat history for the OpenAI API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by OpenAI's API

        Raises:
            KeyError: If the user message format is invalid
        """
        try:
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add chat history if provided
            if chat_hist:
                messages.extend(chat_hist)

            # Add user message
            messages.append({"role": "user", "content": user_message})

            return messages

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def num_tokens_from_string(
        self, model_name: str = "gpt-3.5-turbo", string: str = None
    ) -> int:
        """Returns the number of tokens in a text string.

        Args:
        model_name (str): The tiktoken tokenizer for specifed model name
        string (str): String to calculate the tokens for

        Returns:
        int: Number of tokens"""
        if string and model_name:
            encoding = self.tiktoken.encoding_for_model(model_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        else:
            return 0
