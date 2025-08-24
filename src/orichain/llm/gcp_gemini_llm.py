from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Generator,
    AsyncGenerator,
)
from fastapi import Request
from orichain import error_explainer


class Generate(object):
    """
    Synchronous wrapper for Google's API client for Gemini.

    This class provides methods for generating responses from Google's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Google client for Gemini and set up API key.

        Args:
            - api_key (str): Gemini API key
            - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
            - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from google.genai import Client, types
        from google.genai.client import DebugConfig

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
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
            api_key=kwds.get("api_key"),
            http_options=kwds.get("http_options", types.HttpOptions(timeout=7000)),
            debug_config=kwds.get("debug_config"),
        )

        self.fields = [
            "content",
            "video_metadata",
            "thought",
            "inline_data",
            "code_execution_result",
            "executable_code",
            "file_data",
            "function_call",
            "function_response",
        ]

        self.types = types

    def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Union[Dict[str, str], Any]]],
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the Gemini model to use
            - user_message (Union[str, List[Union[Dict[str, str], Any]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation by GenerateContentConfig
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
                chat_hist=chat_hist,
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Checking tool_choice and formatting tool_config
            tool_config = None
            if tool_choice:
                if tool_choice == "required":
                    tool_choice = "any"
                if tool_choice in ["none", "auto", "any"]:
                    tool_config = self.types.ToolConfig(
                        function_calling_config=self.types.FunctionCallingConfig(
                            mode=tool_choice.upper()
                        )
                    )
                elif tool_choice in [tool.get("name") for tool in tools]:
                    tool_config = self.types.ToolConfig(
                        function_calling_config=self.types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[tool_choice]
                        )
                    )
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }

            # Create new chat session with Google API with the formatted messages
            chat_session = self.client.chats.create(
                model=model_name,
                config=kwds.get(
                    "config",
                    self.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type=(
                            "application/json"
                            if do_json
                            else kwds.get("response_mime_type") or "text/plain"
                        ),
                        tools=[self.types.Tool(function_declarations=tools)]
                        if tools
                        else [],
                        tool_config=tool_config,
                        **sampling_paras,
                    ),
                ),
                history=messages,
            )

            response = chat_session.send_message(message=user_message)

            # Fetching responses from the LLM for tools and text
            result = {
                "response": response.text or "",
                "metadata": {"usage": response.usage_metadata.to_json_dict()},
            }

            if tools:
                tool_calls = []
                if response.function_calls:
                    for tool in response.function_calls:
                        tool_calls.append(
                            {
                                "id": tool.id,
                                "function": {"name": tool.name, "arguments": tool.args},
                            }
                        )
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}

    def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Union[Dict[str, str], Any]]],
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Generator:
        """
        Stream responses from the specified model.

        Args:
            - model_name (str): Name of the Google model to use
            - user_message (Union[str, List[Union[Dict[str, str], Any]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation by GenerateContentConfig
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Yields:
            AsyncGenerator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                chat_hist=chat_hist,
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages

            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Checking tool_choice and formatting tool_config
                tool_config = None
                if tool_choice:
                    if tool_choice == "required":
                        tool_choice = "any"
                    if tool_choice in ["none", "auto", "any"]:
                        tool_config = self.types.ToolConfig(
                            function_calling_config=self.types.FunctionCallingConfig(
                                mode=tool_choice.upper()
                            )
                        )
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        tool_config = self.types.ToolConfig(
                            function_calling_config=self.types.FunctionCallingConfig(
                                mode="ANY", allowed_function_names=[tool_choice]
                            )
                        )
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools."
                        )

                # Create new chat session with Google API with the formatted messages
                chat_session = self.client.chats.create(
                    model=model_name,
                    config=kwds.get(
                        "config",
                        self.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type=(
                                "application/json"
                                if do_json
                                else kwds.get("response_mime_type") or "text/plain"
                            ),
                            tools=[self.types.Tool(function_declarations=tools)]
                            if tools
                            else [],
                            tool_config=tool_config,
                            **sampling_paras,
                        ),
                    ),
                    history=messages,
                )

                result: Dict = {"response": ""}
                tool_calls = []

                for chunk in chat_session.send_message_stream(message=user_message):
                    if chunk.text:
                        result["response"] = result["response"] + chunk.text
                        yield chunk.text
                    elif chunk.function_calls:
                        for tool in chunk.function_calls:
                            tool_calls.append(
                                {
                                    "id": tool.id,
                                    "function": {
                                        "name": tool.name,
                                        "arguments": tool.args,
                                    },
                                }
                            )
                    if chunk.usage_metadata:
                        result.update({"usage": chunk.usage_metadata.to_json_dict()})

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    def _chat_formatter(
        self,
        chat_hist: Optional[List[Union[Dict[str, Union[str, Dict, Any]], Any]]] = None,
    ) -> Union[List[Union[Dict, Any]], Dict, None]:
        """
        Format user messages and chat history for the Google API.

        Args:
            chat_hist (Optional[List[Union[Dict[str, Union[str, Dict, Any]], types.Content]]], optional): Previous conversation history

        Returns:
            List[Dict]: Formatted messages in the structure expected by Google's API
        """
        try:
            # Add chat history if provided
            if chat_hist:
                messages = []
                parts = []
                for chat in chat_hist:
                    if isinstance(chat, Dict):
                        if chat.get("role") == "tool":
                            messages.append(chat)
                        else:
                            for field in self.fields:
                                message = self.types.Part()
                                if field in chat:
                                    setattr(
                                        message,
                                        field if field != "content" else "text",
                                        chat[field],
                                    )
                                parts.append(message)

                            content = self.types.Content(
                                role="user" if chat.get("role") == "user" else "model",
                                parts=parts,
                            )
                            messages.append(content)
                    else:
                        messages.append(chat)
                return messages
            else:
                None

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}


class AsyncGenerate(object):
    """
    Asynchronous wrapper for Google's API client for Gemini.

    This class provides methods for generating responses from Google's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds) -> None:
        """
        Initialize Google client for Gemini and set up API key.

        Args:
            - api_key (str): Gemini API key
            - http_options (types.HttpOptions, optional): HTTP options to be used in each of the requests. Default is None
            - debug_config (DebugConfig, optional): Configuration options that change client network behavior when testing. Default is None

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from google.genai import Client, types
        from google.genai.client import DebugConfig

        # Validate input parameters
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
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
            api_key=kwds.get("api_key"),
            http_options=kwds.get("http_options", types.HttpOptions(timeout=7000)),
            debug_config=kwds.get("debug_config"),
        )

        self.fields = [
            "content",
            "video_metadata",
            "thought",
            "inline_data",
            "code_execution_result",
            "executable_code",
            "file_data",
            "function_call",
            "function_response",
        ]

        self.types = types

    async def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Union[Dict[str, str], Any]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Dict:
        """
        Generate a response from the specified model.

        Args:
            - model_name (str): Name of the Gemini model to use
            - user_message (Union[str, List[Union[Dict[str, str], Any]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation by GenerateContentConfig
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
                chat_hist=chat_hist,
            )

            # Return early if message formatting failed
            if isinstance(messages, Dict):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Checking tool_choice and formatting tool_config
            tool_config = None
            if tool_choice:
                if tool_choice == "required":
                    tool_choice = "any"
                if tool_choice in ["none", "auto", "any"]:
                    tool_config = self.types.ToolConfig(
                        function_calling_config=self.types.FunctionCallingConfig(
                            mode=tool_choice.upper()
                        )
                    )
                elif tool_choice in [tool.get("name") for tool in tools]:
                    tool_config = self.types.ToolConfig(
                        function_calling_config=self.types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[tool_choice]
                        )
                    )
                else:
                    return {
                        "error": 400,
                        "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools.",
                    }

            # Check if the request was disconnected
            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            # Create new chat session with Google API with the formatted messages
            chat_session = self.client.aio.chats.create(
                model=model_name,
                config=kwds.get(
                    "config",
                    self.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type=(
                            "application/json"
                            if do_json
                            else kwds.get("response_mime_type") or "text/plain"
                        ),
                        tools=[self.types.Tool(function_declarations=tools)]
                        if tools
                        else [],
                        tool_config=tool_config,
                        **sampling_paras,
                    ),
                ),
                history=messages,
            )

            response = await chat_session.send_message(message=user_message)

            # Fetching responses from the LLM for tools and text
            result = {
                "response": response.text or "",
                "metadata": {"usage": response.usage_metadata.to_json_dict()},
            }

            if tools:
                tool_calls = []
                if response.function_calls:
                    for tool in response.function_calls:
                        tool_calls.append(
                            {
                                "id": tool.id,
                                "function": {"name": tool.name, "arguments": tool.args},
                            }
                        )
                result["tools"] = tool_calls

            return result

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}

    async def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Union[Dict[str, str], Any]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> AsyncGenerator:
        """
        Stream responses from the specified model.

        Args:
            - model_name (str): Name of the Google model to use
            - user_message (Union[str, List[Union[Dict[str, str], Any]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation by GenerateContentConfig
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "none" for no tools, "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Yields:
            AsyncGenerator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                chat_hist=chat_hist,
            )

            # Yield error and return early if message formatting failed
            if isinstance(messages, Dict):
                yield messages

            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Checking tool_choice and formatting tool_config
                tool_config = None
                if tool_choice:
                    if tool_choice == "required":
                        tool_choice = "any"
                    if tool_choice in ["none", "auto", "any"]:
                        tool_config = self.types.ToolConfig(
                            function_calling_config=self.types.FunctionCallingConfig(
                                mode=tool_choice.upper()
                            )
                        )
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        tool_config = self.types.ToolConfig(
                            function_calling_config=self.types.FunctionCallingConfig(
                                mode="ANY", allowed_function_names=[tool_choice]
                            )
                        )
                    else:
                        raise ValueError(
                            f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['none', 'auto', 'required'] or match a tool name in the provided tools."
                        )

                # Create new chat session with Google API with the formatted messages
                chat_session = self.client.aio.chats.create(
                    model=model_name,
                    config=kwds.get(
                        "config",
                        self.types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type=(
                                "application/json"
                                if do_json
                                else kwds.get("response_mime_type") or "text/plain"
                            ),
                            tools=[self.types.Tool(function_declarations=tools)]
                            if tools
                            else [],
                            tool_config=tool_config,
                            **sampling_paras,
                        ),
                    ),
                    history=messages,
                )

                result: Dict = {"response": ""}
                tool_calls = []

                async for chunk in await chat_session.send_message_stream(
                    message=user_message
                ):
                    if request and await request.is_disconnected():
                        yield {"error": 400, "reason": "request aborted by user"}
                        break

                    if chunk.text:
                        result["response"] = result["response"] + chunk.text
                        yield chunk.text
                    elif chunk.function_calls:
                        for tool in chunk.function_calls:
                            tool_calls.append(
                                {
                                    "id": tool.id,
                                    "function": {
                                        "name": tool.name,
                                        "arguments": tool.args,
                                    },
                                }
                            )
                    if chunk.usage_metadata:
                        result.update({"usage": chunk.usage_metadata.to_json_dict()})

                if tools:
                    result["tools"] = tool_calls

                yield result
        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    async def _chat_formatter(
        self,
        chat_hist: Optional[List[Union[Dict[str, Union[str, Dict, Any]], Any]]] = None,
    ) -> Union[List[Union[Dict, Any]], Dict, None]:
        """
        Format user messages and chat history for the Google API.

        Args:
            chat_hist (Optional[List[Union[Dict[str, Union[str, Dict, Any]], types.Content]]], optional): Previous conversation history

        Returns:
            List[Dict]: Formatted messages in the structure expected by Google's API
        """
        try:
            # Add chat history if provided
            if chat_hist:
                messages = []
                parts = []
                for chat in chat_hist:
                    if isinstance(chat, Dict):
                        if chat.get("role") == "tool":
                            messages.append(chat)
                        else:
                            for field in self.fields:
                                message = self.types.Part()
                                if field in chat:
                                    setattr(
                                        message,
                                        field if field != "content" else "text",
                                        chat[field],
                                    )
                                parts.append(message)

                            content = self.types.Content(
                                role="user" if chat.get("role") == "user" else "model",
                                parts=parts,
                            )
                            messages.append(content)
                    else:
                        messages.append(chat)
                return messages
            else:
                None

        except Exception as e:
            error_explainer(e=e)
            return {"error": 500, "reason": str(e)}
