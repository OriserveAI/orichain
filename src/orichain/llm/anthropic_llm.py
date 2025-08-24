from typing import Any, List, Dict, Optional, Union, Generator, AsyncGenerator

from fastapi import Request

from orichain import error_explainer


class Generate(object):
    """
    Synchronous wrapper for Anthropic's API client.

    This class provides methods for generating responses from Anthropic's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize Anthropic client and set up API key.

        Args:
            - api_key (str): Anthropic API key
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default is 2
            - prompt_caching (bool, optional): Whether to use prompt caching. Default: True

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """
        from httpx import Timeout

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required 'api_key' not found")
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

        if kwds.get("prompt_caching", True):
            self.prompt_caching = True
        else:
            self.prompt_caching = False

        from anthropic import Anthropic, NOT_GIVEN

        # Initialize the Anthropic client with provided parameters
        self.client = Anthropic(
            api_key=kwds.get("api_key"),
            timeout=kwds.get(
                "timeout", Timeout(60.0, read=5.0, write=10.0, connect=2.0)
            ),
            max_retries=kwds.get("max_retries", 2),
        )
        self.not_given = NOT_GIVEN

    def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: bool = False,
        **kwds: Any,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the Anthropic model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Setting system prompt in sampling params, as None type is not allowed
            if system_prompt:
                system = [{"type": "text", "text": system_prompt}]
                if self.prompt_caching:
                    system[0].update({"cache_control": {"type": "ephemeral"}})
                sampling_paras["system"] = system

            # Setting default max_tokens if not provided
            if "max_tokens" not in sampling_paras:
                sampling_paras["max_tokens"] = 512

            # Check if tools and tool_choice are provided and format them
            tools = (
                [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["parameters"],
                    }
                    for tool in tools
                ]
                if tools
                else []
            )
            if self.prompt_caching and tools:
                tools[-1].update({"cache_control": {"type": "ephemeral"}})
            if tool_choice:
                if tool_choice in ["none", "auto"]:
                    tool_choice = {"type": tool_choice}
                elif tool_choice == "required":
                    tool_choice = {"type": "any"}
                elif tool_choice in [tool.get("name") for tool in tools]:
                    tool_choice = {"type": "tool", "name": tool_choice}
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }
            else:
                tool_choice = self.not_given

            # Call the Anthropic API with the formatted messages
            message = self.client.with_options(
                timeout=kwds.get("timeout")
            ).messages.create(
                messages=messages,
                model=model_name,
                tools=tools,
                tool_choice=tool_choice,
                **sampling_paras,
            )

            # Format the response with metadata
            result = {"response": "", "metadata": {"usage": message.usage.to_dict()}}
            tool_calls = []

            for content in message.content:
                if content.type == "text":
                    result["response"] += (
                        "{" + content.text if do_json else content.text
                    )
                elif content.type == "tool_use":
                    tool = content.to_dict()
                    tool["function"] = {
                        "name": tool.pop("name"),
                        "arguments": tool.pop("input"),
                    }
                    tool_calls.append(tool)

            if tools:
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
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
            - model_name (str): Name of the Anthropic model to use
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
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages
            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Setting system prompt in sampling params, as None type is not allowed
                if system_prompt:
                    system = [{"type": "text", "text": system_prompt}]
                    if self.prompt_caching:
                        system[0].update({"cache_control": {"type": "ephemeral"}})
                    sampling_paras["system"] = system

                # Setting default max_tokens if not provided
                if "max_tokens" not in sampling_paras:
                    sampling_paras["max_tokens"] = 512

                # Check if tools and tool_choice are provided and format them
                tools = (
                    [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "input_schema": tool["parameters"],
                        }
                        for tool in tools
                    ]
                    if tools
                    else []
                )
                if self.prompt_caching and tools:
                    tools[-1].update({"cache_control": {"type": "ephemeral"}})
                if tool_choice:
                    if tool_choice in ["none", "auto"]:
                        tool_choice = {"type": tool_choice}
                    elif tool_choice == "required":
                        tool_choice = {"type": "any"}
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        tool_choice = {"type": "tool", "name": tool_choice}
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools."
                        )
                else:
                    tool_choice = self.not_given

                # Start the streaming session
                with self.client.messages.stream(
                    messages=messages,
                    model=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    **sampling_paras,
                ) as stream:
                    # Start JSON response if requested
                    if do_json:
                        yield "{"

                    # Stream text chunks as they become available
                    for chunk in stream.text_stream:
                        # Yield non-empty chunks
                        if chunk:
                            yield chunk

                # Get the final complete message after streaming
                final_response = stream.get_final_message()

                # Format the final response with metadata
                result = {
                    "response": "",
                    "metadata": {"usage": final_response.usage.to_dict()},
                }
                tool_calls = []
                for content in final_response.content:
                    if content.type == "text":
                        result["response"] += (
                            "{" + content.text if do_json else content.text
                        )
                    elif content.type == "tool_use":
                        tool = content.to_dict()
                        tool["function"] = {
                            "name": tool.pop("name"),
                            "arguments": tool.pop("input"),
                        }
                        tool_calls.append(tool)

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    def _chat_formatter(
        self,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
        do_json: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Format user messages and chat history for the Anthropic API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by Anthropic's API

        Raises:
            KeyError: If the user message format is invalid
        """
        try:
            messages = []

            # Add chat history if provided
            if chat_hist:
                messages.extend(chat_hist)

            # Add user message based on its type
            if isinstance(user_message, str):
                content = [{"type": "text", "text": user_message}]
                if self.prompt_caching:
                    content[0].update({"cache_control": {"type": "ephemeral"}})
                messages.append({"role": "user", "content": content})
            elif isinstance(user_message, List):
                messages.extend(user_message)
            else:
                raise KeyError(
                    """invalid user messages format, this error should not happen due to the validation steps being used.\
                    Please check the validation function"""
                )

            # Add JSON formatting instruction if requested
            if do_json:
                messages.append({"role": "assistant", "content": "{"})

            return messages

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncGenerate(object):
    """
    Asynchronous wrapper for Anthropic's API client.

    This class provides methods for generating responses from Anthropic's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize Anthropic client and set up API key.

        Args:
            - api_key (str): Anthropic API key
            - timeout (Timeout, optional): Request timeout parameter like connect, read, write. Default is 60.0, 5.0, 10.0, 2.0
            - max_retries (int, optional): Number of retries for the request. Default is 2
            - prompt_caching (bool, optional): Whether to use prompt caching. Default: True

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """
        from httpx import Timeout

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required 'api_key' not found")
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

        if kwds.get("prompt_caching", True):
            self.prompt_caching = True
        else:
            self.prompt_caching = False

        from anthropic import AsyncAnthropic, NOT_GIVEN

        # Initialize the Anthropic client with provided parameters
        self.client = AsyncAnthropic(
            api_key=kwds.get("api_key"),
            timeout=kwds.get(
                "timeout", Timeout(60.0, read=5.0, write=10.0, connect=2.0)
            ),
            max_retries=kwds.get("max_retries", 2),
        )
        self.not_given = NOT_GIVEN

    async def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: bool = False,
        **kwds: Any,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the Anthropic model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Setting system prompt in sampling params, as None type is not allowed
            if system_prompt:
                system = [{"type": "text", "text": system_prompt}]
                if self.prompt_caching:
                    system[0].update({"cache_control": {"type": "ephemeral"}})
                sampling_paras["system"] = system

            # Setting default max_tokens if not provided
            if "max_tokens" not in sampling_paras:
                sampling_paras["max_tokens"] = 512

            # Check if tools and tool_choice are provided and format them
            tools = (
                [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "input_schema": tool["parameters"],
                    }
                    for tool in tools
                ]
                if tools
                else []
            )
            if self.prompt_caching and tools:
                tools[-1].update({"cache_control": {"type": "ephemeral"}})
            if tool_choice:
                if tool_choice in ["none", "auto"]:
                    tool_choice = {"type": tool_choice}
                elif tool_choice == "required":
                    tool_choice = {"type": "any"}
                elif tool_choice in [tool.get("name") for tool in tools]:
                    tool_choice = {"type": "tool", "name": tool_choice}
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }
            else:
                tool_choice = self.not_given

            # Check if the request was disconnected
            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            # Call the Anthropic API with the formatted messages
            message = await self.client.with_options(
                timeout=kwds.get("timeout")
            ).messages.create(
                messages=messages,
                model=model_name,
                tools=tools,
                tool_choice=tool_choice,
                **sampling_paras,
            )

            # Format the response with metadata
            result = {"response": "", "metadata": {"usage": message.usage.to_dict()}}
            tool_calls = []

            for content in message.content:
                if content.type == "text":
                    result["response"] += (
                        "{" + content.text if do_json else content.text
                    )
                elif content.type == "tool_use":
                    tool = content.to_dict()
                    tool["function"] = {
                        "name": tool.pop("name"),
                        "arguments": tool.pop("input"),
                    }
                    tool_calls.append(tool)

            if tools:
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
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
            - model_name (str): Name of the Anthropic model to use
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
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages
            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Setting system prompt in sampling params, as None type is not allowed
                if system_prompt:
                    system = [{"type": "text", "text": system_prompt}]
                    if self.prompt_caching:
                        system[0].update({"cache_control": {"type": "ephemeral"}})
                    sampling_paras["system"] = system

                # Setting default max_tokens if not provided
                if "max_tokens" not in sampling_paras:
                    sampling_paras["max_tokens"] = 512

                # Check if tools and tool_choice are provided and format them
                tools = (
                    [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "input_schema": tool["parameters"],
                        }
                        for tool in tools
                    ]
                    if tools
                    else []
                )
                if self.prompt_caching and tools:
                    tools[-1].update({"cache_control": {"type": "ephemeral"}})
                if tool_choice:
                    if tool_choice in ["none", "auto"]:
                        tool_choice = {"type": tool_choice}
                    elif tool_choice == "required":
                        tool_choice = {"type": "any"}
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        tool_choice = {"type": "tool", "name": tool_choice}
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools."
                        )
                else:
                    tool_choice = self.not_given

                # Start the streaming session
                async with self.client.messages.stream(
                    messages=messages,
                    model=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    **sampling_paras,
                ) as stream:
                    # Start JSON response if requested
                    if do_json:
                        yield "{"

                    # Stream text chunks as they become available
                    async for chunk in stream.text_stream:
                        # Check if the request was disconnected
                        if request and await request.is_disconnected():
                            yield {"error": 400, "reason": "request aborted by user"}
                            await stream.close()
                            break

                        # Yield non-empty chunks
                        if chunk:
                            yield chunk

                # Get the final complete message after streaming
                final_response = await stream.get_final_message()

                # Format the final response with metadata
                result = {
                    "response": "",
                    "metadata": {"usage": final_response.usage.to_dict()},
                }
                tool_calls = []
                for content in final_response.content:
                    if content.type == "text":
                        result["response"] += (
                            "{" + content.text if do_json else content.text
                        )
                    elif content.type == "tool_use":
                        tool = content.to_dict()
                        tool["function"] = {
                            "name": tool.pop("name"),
                            "arguments": tool.pop("input"),
                        }
                        tool_calls.append(tool)

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    async def _chat_formatter(
        self,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
        do_json: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Format user messages and chat history for the Anthropic API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by Anthropic's API

        Raises:
            KeyError: If the user message format is invalid
        """
        try:
            messages = []

            # Add chat history if provided
            if chat_hist:
                messages.extend(chat_hist)

            # Add user message based on its type
            if isinstance(user_message, str):
                content = [{"type": "text", "text": user_message}]
                if self.prompt_caching:
                    content[0].update({"cache_control": {"type": "ephemeral"}})
                messages.append({"role": "user", "content": content})
            elif isinstance(user_message, List):
                messages.extend(user_message)
            else:
                raise KeyError(
                    """invalid user messages format, this error should not happen due to the validation steps being used.\
                    Please check the validation function"""
                )

            # Add JSON formatting instruction if requested
            if do_json:
                messages.append({"role": "assistant", "content": "{"})

            return messages

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
