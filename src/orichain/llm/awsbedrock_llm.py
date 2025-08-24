from typing import Any, List, Dict, Optional, Union, Generator, AsyncGenerator
from botocore.eventstream import EventStream
import asyncio
import json
from fastapi import Request
from orichain import error_explainer


class CreateAiter(object):
    """
    Asynchronous iterator wrapper to wrap synchronous iterator
    NOTE: This is currently only for the use of awsbedrock converse stream in async method
    """

    def __init__(self, event_stream: EventStream) -> None:
        """
        Convert EventStream(AWS) into a iterator
        """
        self.sync_streamer = iter(event_stream)
        self.SENTINEL = object()

    def __aiter__(self):
        return self

    async def __anext__(self):
        """
        Returns the next event stream asynchronously.
        """
        try:
            stream_output = next(self.sync_streamer)
        except StopIteration:
            stream_output = self.SENTINEL
        except Exception as e:
            error_explainer(e)
            stream_output = {"error": 500, "reason": str(e)}

        return stream_output


class Generate(object):
    """
    Synchronous wrapper for AWS Bedrock's API client.

    This class provides methods for generating responses from AWS Bedrock's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize AWSBedrock client and set up API keys.

        Args:
            - aws_access_key (str): access key
            - aws_secret_key (str): api key
            - aws_region (str): region name
            - config (Config, optional):
                - connect_timeout (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to make a connection. Default: 60
                - read_timeout: (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to read from a connection. Default: 60
                - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                - retries (Dict, optional):
                    - total_max_attempts: Number of retries for the request. Default: 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from botocore.config import Config

        # Validate input parameters
        if not kwds.get("aws_access_key"):
            raise KeyError("Required 'aws_access_key' not found")
        elif not kwds.get("aws_secret_key"):
            raise KeyError("Required 'aws_secret_key' not found")
        elif not kwds.get("aws_region"):
            raise KeyError("Required aws_region not found")
        elif kwds.get("config") and not isinstance(kwds.get("config"), Config):
            raise TypeError(
                "Invalid 'config' type detected:",
                type(kwds.get("config")),
                ", Please enter valid config using:\n'from botocore.config import Config'",
            )
        else:
            pass

        if kwds.get("prompt_caching", True):
            self.prompt_caching = True
        else:
            self.prompt_caching = False

        import boto3

        # Initialize the AWSBedock boto client with provided parameters
        self.client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=kwds.get("aws_access_key"),
            aws_secret_access_key=kwds.get("aws_secret_key"),
            config=kwds.get("config")
            or Config(
                region_name=kwds.get("aws_region"),
                read_timeout=10,
                connect_timeout=2,
                retries={"total_max_attempts": 2},
                max_pool_connections=100,
            ),
        )

    def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
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
            - model_name (str): Name of the AWS Bedrock model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            # Return early if message formatting failed
            if not isinstance(messages, List):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Setting up the request body
            body = {
                "modelId": model_name,
                "messages": messages,
                "inferenceConfig": sampling_paras,
                "additionalModelRequestFields": kwds.get("additional_model_fields", {}),
            }

            # Check if tools and tool_choice are provided and format them
            if tools:
                body["toolConfig"] = {"tools": []}
                for tool in tools:
                    body["toolConfig"]["tools"].append(
                        {
                            "toolSpec": {
                                "name": tool.get("name"),
                                "description": tool.get("description"),
                                "inputSchema": {"json": tool.get("parameters", {})},
                            },
                        }
                    )
                if tool_choice:
                    if tool_choice == "auto":
                        body["toolConfig"]["toolChoice"] = {"auto": {}}
                    elif tool_choice == "required":
                        body["toolConfig"]["toolChoice"] = {"any": {}}
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        body["toolConfig"]["toolChoice"] = {
                            "tool": {"name": tool_choice}
                        }
                    else:
                        return {
                            "error": 400,
                            "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools.",
                        }

            # Check for system_prompt
            if system_prompt:
                system = [{"text": system_prompt}]
                if self.prompt_caching:
                    system.append({"cachePoint": {"type": "default"}})
                body.update({"system": system})

            # Call the AWSBedrock client with the formatted messages
            result = self._generate_response(body=body)

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
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
            - model_name (str): Name of the AWS Bedrock model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Yields:
            Generator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            # Yield error and return early if message formatting failed
            if not isinstance(messages, List):
                yield messages
            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Setting up the request body
                body = {
                    "modelId": model_name,
                    "messages": messages,
                    "inferenceConfig": sampling_paras,
                    "additionalModelRequestFields": kwds.get(
                        "additional_model_fields", {}
                    ),
                }

                # Check if tools are provided and format them
                if tools:
                    body["toolConfig"] = {"tools": []}
                    for tool in tools:
                        body["toolConfig"]["tools"].append(
                            {
                                "toolSpec": {
                                    "name": tool.get("name"),
                                    "description": tool.get("description"),
                                    "inputSchema": {"json": tool.get("parameters", {})},
                                },
                            }
                        )
                    if tool_choice:
                        if tool_choice == "auto":
                            body["toolConfig"]["toolChoice"] = {"auto": {}}
                        elif tool_choice == "required":
                            body["toolConfig"]["toolChoice"] = {"any": {}}
                        elif tool_choice in [tool.get("name") for tool in tools]:
                            body["toolConfig"]["toolChoice"] = {
                                "tool": {"name": tool_choice}
                            }
                        else:
                            raise ValueError(
                                f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools."
                            )

                # Check for system_prompt
                if system_prompt:
                    system = [{"text": system_prompt}]
                    if self.prompt_caching:
                        system.append({"cachePoint": {"type": "default"}})
                    body.update({"system": system})

                # Start the streaming session
                streaming_response = self._stream_response(body=body)

                response = ""
                tool_calls = []
                usage = None
                no_error = True

                # Stream text chunks as they become available
                for text in streaming_response:
                    if text and isinstance(text, str):
                        response += text
                        yield text
                    elif isinstance(text, Dict):
                        if text.get("toolUseId"):
                            tool = text
                            tool["id"] = tool.pop("toolUseId")
                            tool["function"] = {
                                "name": tool.pop("name"),
                                "arguments": tool.pop("input", {}),
                            }
                            tool_calls.append(tool)
                        elif tool_args := text.get("input"):
                            tool_calls[-1]["function"]["arguments"] = json.loads(
                                tool_args
                            )
                        elif "error" not in text:
                            # Final chunk from AWS Bedrock while streaming
                            usage = text
                        elif "error" in text:
                            no_error = False
                            streaming_response.close()

                            # Yield non-empty chunks
                            yield text

                            break
                        else:
                            pass
                    else:
                        pass

                if no_error:
                    # Format the final response with metadata
                    result = {
                        "response": response.strip(),
                        "metadata": {"usage": usage},
                    }

                    if tools:
                        result["tools"] = tool_calls

                    yield result

        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    def _generate_response(self, body: Dict) -> Dict:
        """Converse function for generating response

        Args:
            body (Dict): Contains all the paras to pass

        Returns:
            Dict: Formatted response from the Converse"""
        try:
            # Call to Bedrock service from Converse method
            response = self.client.converse(**body)

            # Structuring response
            result = {"response": ""}
            content = response.get("output", {}).get("message", {}).get("content", [{}])
            if body.get("toolConfig"):
                result["tools"] = []
                for cont in content:
                    if cont.get("text"):
                        result["response"] = cont.get("text", "").strip()
                    elif tool := cont.get("toolUse"):
                        tool["id"] = tool.pop("toolUseId")
                        tool["function"] = {
                            "name": tool.pop("name"),
                            "arguments": tool.pop("input", {}),
                        }
                        result["tools"].append(tool)
            else:
                result["response"] = content[0].get("text", "").strip()

            result["metadata"] = {"usage": response.get("usage", {})}

            if response.get("metrics"):
                result["metadata"]["usage"].update(response.get("metrics"))

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def _stream_response(self, body: Dict) -> Generator:
        """ConverseStream function for generating response

        Args:
            body (Dict): Contains all the paras to pass

        Yeilds:
            Generator: Chunks of the model's response or error information"""
        try:
            # Call to Bedrock service from ConverseStream method
            response = self.client.converse_stream(**body)

            # Fetching generator
            streaming_response = response.get("stream")

            # Start the streaming session
            for event in streaming_response:
                # Waiting for text chunks to be generated
                if (
                    text := event.get("contentBlockDelta", {})
                    .get("delta", {})
                    .get("text")
                ):
                    yield text
                elif (
                    tool := event.get("contentBlockStart", {})
                    .get("start", {})
                    .get("toolUse")
                ):
                    yield tool
                elif (
                    tool_args := event.get("contentBlockDelta", {})
                    .get("delta", {})
                    .get("toolUse")
                ):
                    yield tool_args
                elif event.get("metadata", {}).get("usage"):
                    usage = event.get("metadata").get("usage")

                    if event.get("metadata").get("metrics"):
                        usage.update(event.get("metadata").get("metrics"))

                    yield usage

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
        Format user messages and chat history for the AWS Bedrock API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by AWS Bedrock's API

        Raises:
            KeyError: If the user message format is invalid

        NOTE: JSON METHOD UNSTABLE
        """
        try:
            messages = []

            # Add chat history if provided
            if chat_hist:
                for chat_log in chat_hist:
                    messages.append(
                        {
                            "role": chat_log.get("role"),
                            "content": [{"text": chat_log.get("content")}],
                        }
                    )

            # Add user message based on its type
            if isinstance(user_message, str):
                content = [
                    {
                        "text": user_message
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                        if do_json
                        else user_message
                    }
                ]
                if self.prompt_caching:
                    content.append({"cachePoint": {"type": "default"}})
                messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
            elif isinstance(user_message, List):
                for logs in user_message:
                    messages.append(
                        {
                            "role": logs.get("role"),
                            "content": [{"text": logs.get("content")}],
                        }
                    )

                if messages[-1].get("role") == "user" and do_json:
                    messages[-1]["content"][0]["text"] = (
                        messages[-1]["content"][0]["text"]
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                    )
                elif messages[-1].get("role") == "assistant" and do_json:
                    messages[-2]["content"][0]["text"] = (
                        messages[-2]["content"][0]["text"]
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                    )
                else:
                    pass

            else:
                pass

            return messages

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncGenerate(object):
    """
    Asynchronous wrapper for AWS Bedrock's API client.

    This class provides methods for generating responses from AWS Bedrock's models
    both in streaming and non-streaming modes. It handles chat history formatting,
    error handling, and proper request configuration.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize AWSBedrock client and set up API keys.

        Args:
            - aws_access_key (str): access key
            - aws_secret_key (str): api key
            - aws_region (str): region name
            - config (Config, optional):
                - connect_timeout (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to make a connection. Default: 60
                - read_timeout: (float or int, optional): The time in seconds till a timeout exception is thrown when attempting to read from a connection. Default: 60
                - region_name (str, optional): region name Note: If specifing config you need to still pass region_name even if you have already passed in aws_region
                - max_pool_connections: The maximum number of connections to keep in a connection pool. Defualt: 10
                - retries (Dict, optional):
                    - total_max_attempts: Number of retries for the request. Default: 2

        Raises:
            - KeyError: If required parameters are not provided.
            - TypeError: If an invalid type is provided for a parameter
        """

        from botocore.config import Config

        # Validate input parameters
        if not kwds.get("aws_access_key"):
            raise KeyError("Required 'aws_access_key' not found")
        elif not kwds.get("aws_secret_key"):
            raise KeyError("Required 'aws_secret_key' not found")
        elif not kwds.get("aws_region"):
            raise KeyError("Required aws_region not found")
        elif kwds.get("config") and not isinstance(kwds.get("config"), Config):
            raise TypeError(
                "Invalid 'config' type detected:",
                type(kwds.get("config")),
                ", Please enter valid config using:\n'from botocore.config import Config'",
            )
        else:
            pass

        if kwds.get("prompt_caching", True):
            self.prompt_caching = True
        else:
            self.prompt_caching = False

        import boto3

        # Initialize the AWSBedock boto client with provided parameters
        self.client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=kwds.get("aws_access_key"),
            aws_secret_access_key=kwds.get("aws_secret_key"),
            config=kwds.get("config")
            or Config(
                region_name=kwds.get("aws_region"),
                read_timeout=10,
                connect_timeout=2,
                retries={"total_max_attempts": 2},
                max_pool_connections=100,
            ),
        )

    async def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
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
            - model_name (str): Name of the AWS Bedrock model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Returns:
            Dict: Response from the model or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            # Return early if message formatting failed
            if not isinstance(messages, List):
                return messages

            # Default empty dictionaries
            sampling_paras = sampling_paras or {}

            # Check if the request was disconnected
            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            # Setting up the request body
            body = {
                "modelId": model_name,
                "messages": messages,
                "inferenceConfig": sampling_paras,
                "additionalModelRequestFields": kwds.get("additional_model_fields", {}),
            }

            # Check if tools are provided and format them
            if tools:
                body["toolConfig"] = {"tools": []}
                for tool in tools:
                    body["toolConfig"]["tools"].append(
                        {
                            "toolSpec": {
                                "name": tool.get("name"),
                                "description": tool.get("description"),
                                "inputSchema": {"json": tool.get("parameters", {})},
                            },
                        }
                    )
                if tool_choice:
                    if tool_choice == "auto":
                        body["toolConfig"]["toolChoice"] = {"auto": {}}
                    elif tool_choice == "required":
                        body["toolConfig"]["toolChoice"] = {"any": {}}
                    elif tool_choice in [tool.get("name") for tool in tools]:
                        body["toolConfig"]["toolChoice"] = {
                            "tool": {"name": tool_choice}
                        }
                    else:
                        return {
                            "error": 400,
                            "reason": f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools.",
                        }

            # Check for system_prompt
            if system_prompt:
                system = [{"text": system_prompt}]
                if self.prompt_caching:
                    system.append({"cachePoint": {"type": "default"}})
                body.update({"system": system})

            # Call the AWSBedrock client with the formatted messages
            result = await self._generate_response(body=body)

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
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
            - model_name (str): Name of the AWS Bedrock model to use
            - user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            - system_prompt (Optional[str], optional): System prompt to provide context to the model
            - request (Optional[Request], optional): FastAPI request object for connection tracking
            - chat_hist (Optional[List[str]], optional): Previous conversation history
            - sampling_paras (Optional[Dict], optional): Parameters for controlling the model's generation
            - tools (List[Dict], optional): List of tools to be used by the model.
            - tool_choice (Optional[str], optional): Specifies if and which tool the model must call — "auto" for automatic, "required" for mandatory, or a specific tool's name.
            - do_json (bool, optional): Whether to format the response as JSON. Defaults to False
            - **kwds: Additional keyword arguments to pass to the client

        Yields:
            AsyncGenerator: Chunks of the model's response or error information
        """
        try:
            # Format the chat history and user message
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            # Yield error and return early if message formatting failed
            if not isinstance(messages, List):
                yield messages
            else:
                # Default empty dictionaries
                sampling_paras = sampling_paras or {}

                # Setting up the request body
                body = {
                    "modelId": model_name,
                    "messages": messages,
                    "inferenceConfig": sampling_paras,
                    "additionalModelRequestFields": kwds.get(
                        "additional_model_fields", {}
                    ),
                }

                # Check if tools are provided and format them
                if tools:
                    body["toolConfig"] = {"tools": []}
                    for tool in tools:
                        body["toolConfig"]["tools"].append(
                            {
                                "toolSpec": {
                                    "name": tool.get("name"),
                                    "description": tool.get("description"),
                                    "inputSchema": {"json": tool.get("parameters", {})},
                                },
                            }
                        )
                    if tool_choice:
                        if tool_choice == "auto":
                            body["toolConfig"]["toolChoice"] = {"auto": {}}
                        elif tool_choice == "required":
                            body["toolConfig"]["toolChoice"] = {"any": {}}
                        elif tool_choice in [tool.get("name") for tool in tools]:
                            body["toolConfig"]["toolChoice"] = {
                                "tool": {"name": tool_choice}
                            }
                        else:
                            raise ValueError(
                                f"Invalid tool_choice '{tool_choice}' provided. It must be one of ['auto', 'required'] or match a tool name in the provided tools."
                            )

                # Check for system_prompt
                if system_prompt:
                    system = [{"text": system_prompt}]
                    if self.prompt_caching:
                        system.append({"cachePoint": {"type": "default"}})
                    body.update({"system": system})

                # Start the streaming session
                streaming_response = self._stream_response(body=body)

                response = ""
                tool_calls = []
                usage = None
                no_error = True

                # Stream text chunks as they become available
                async for text in streaming_response:
                    # Check if the request was disconnected
                    if request and await request.is_disconnected():
                        yield {"error": 400, "reason": "request aborted by user"}
                        await streaming_response.aclose()
                        break
                    elif text and isinstance(text, str):
                        response += text
                        yield text
                    elif isinstance(text, Dict):
                        if text.get("toolUseId"):
                            tool = text
                            tool["id"] = tool.pop("toolUseId")
                            tool["function"] = {
                                "name": tool.pop("name"),
                                "arguments": tool.pop("input", {}),
                            }
                            tool_calls.append(tool)
                        elif tool_args := text.get("input"):
                            tool_calls[-1]["function"]["arguments"] = json.loads(
                                tool_args
                            )
                        elif "error" not in text:
                            # Final chunk from AWS Bedrock while streaming
                            usage = text
                        elif "error" in text:
                            no_error = False
                            await streaming_response.aclose()

                            # Yield non-empty chunks
                            yield text

                            break
                        else:
                            pass
                    else:
                        pass

                if no_error:
                    # Format the final response with metadata
                    result = {
                        "response": response.strip(),
                        "metadata": {"usage": usage},
                    }

                    if tools:
                        result["tools"] = tool_calls

                    yield result

        except Exception as e:
            error_explainer(e)
            yield {"error": 500, "reason": str(e)}

    async def _generate_response(self, body: Dict) -> Dict:
        """Converse function for generating response

        Args:
            body (Dict): Contains all the paras to pass

        Returns:
            Dict: Formatted response from the Converse"""
        try:
            # Call to Bedrock service from Converse method
            response = await asyncio.to_thread(self.client.converse, **body)

            # Structuring response
            result = {"response": ""}
            content = response.get("output", {}).get("message", {}).get("content", [{}])
            if body.get("toolConfig"):
                result["tools"] = []
                for cont in content:
                    if cont.get("text"):
                        result["response"] = cont.get("text", "").strip()
                    elif tool := cont.get("toolUse"):
                        tool["id"] = tool.pop("toolUseId")
                        tool["function"] = {
                            "name": tool.pop("name"),
                            "arguments": tool.pop("input", {}),
                        }
                        result["tools"].append(tool)
            else:
                result["response"] = content[0].get("text", "").strip()

            result["metadata"] = {"usage": response.get("usage", {})}

            if response.get("metrics"):
                result["metadata"]["usage"].update(response.get("metrics"))

            return result

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def _stream_response(self, body: Dict) -> AsyncGenerator:
        """ConverseStream function for generating response

        Args:
            body (Dict): Contains all the paras to pass

        Yeilds:
            AsyncGenerator: Chunks of the model's response or error information"""
        try:
            # Call to Bedrock service from ConverseStream method
            response = await asyncio.to_thread(self.client.converse_stream, **body)

            # Fetching generator
            streaming_response = CreateAiter(event_stream=response.get("stream"))

            # Use the async wrapper to iterate over events asynchronously.
            async for event in streaming_response:
                # Waiting for text chunks to be generated.
                if event is streaming_response.SENTINEL:
                    break
                elif (
                    text := event.get("contentBlockDelta", {})
                    .get("delta", {})
                    .get("text")
                ):
                    yield text
                elif (
                    tool := event.get("contentBlockStart", {})
                    .get("start", {})
                    .get("toolUse")
                ):
                    yield tool
                elif (
                    tool_args := event.get("contentBlockDelta", {})
                    .get("delta", {})
                    .get("toolUse")
                ):
                    yield tool_args
                elif usage := event.get("metadata", {}).get("usage"):
                    if metrics := event["metadata"].get("metrics"):
                        usage.update(metrics)
                    yield usage
                elif event.get("error"):
                    yield event
                else:
                    pass

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
        Format user messages and chat history for the AWS Bedrock API.

        Args:
            user_message (Union[str, List[Dict[str, str]]]): The user's message or formatted messages
            chat_hist (Optional[List[Dict[str, str]]], optional): Previous conversation history
            do_json (Optional[bool], optional): Whether to format the response as JSON. Defaults to False

        Returns:
            List[Dict]: Formatted messages in the structure expected by AWS Bedrock's API

        Raises:
            KeyError: If the user message format is invalid

        NOTE: JSON METHOD UNSTABLE
        """
        try:
            messages = []

            # Add chat history if provided
            if chat_hist:
                for chat_log in chat_hist:
                    messages.append(
                        {
                            "role": chat_log.get("role"),
                            "content": [{"text": chat_log.get("content")}],
                        }
                    )

            # Add user message based on its type
            if isinstance(user_message, str):
                content = [
                    {
                        "text": user_message
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                        if do_json
                        else user_message
                    }
                ]
                if self.prompt_caching:
                    content.append({"cachePoint": {"type": "default"}})
                messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
            elif isinstance(user_message, List):
                for logs in user_message:
                    messages.append(
                        {
                            "role": logs.get("role"),
                            "content": [{"text": logs.get("content")}],
                        }
                    )

                if messages[-1].get("role") == "user" and do_json:
                    messages[-1]["content"][0]["text"] = (
                        messages[-1]["content"][0]["text"]
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                    )
                elif messages[-1].get("role") == "assistant" and do_json:
                    messages[-2]["content"][0]["text"] = (
                        messages[-2]["content"][0]["text"]
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                    )
                else:
                    pass

            else:
                pass

            return messages

        except Exception as e:
            error_explainer(e)
