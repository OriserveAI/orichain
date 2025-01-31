from typing import Any, List, Dict, Optional, Union, AsyncGenerator
import traceback

from fastapi import Request

import asyncio


class Generate(object):
    def __init__(self, **kwds: Any) -> None:
        """
        Initialize AWS Bedrock client and set up API key.

        Args:
            - aws_access_key (str): access key
            - aws_secret_key (str): api key
            - aws_region (str): region name
        """

        from botocore.config import Config

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

        import boto3

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
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Dict:
        try:
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            if not isinstance(messages, List):
                return messages

            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            body = {
                "modelId": model_name,
                "messages": messages,
                "inferenceConfig": sampling_paras,
                "additionalModelRequestFields": kwds.get("additional_model_fields", {}),
            }

            if system_prompt:
                body.update({"system": [{"text": system_prompt}]})

            result = await self._generate_response(body=body)

            return result

        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}

    async def streaming(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> AsyncGenerator:
        try:
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                do_json=do_json,
            )

            if not isinstance(messages, List):
                yield messages
            else:
                body = {
                    "modelId": model_name,
                    "messages": messages,
                    "inferenceConfig": sampling_paras,
                    "additionalModelRequestFields": kwds.get(
                        "additional_model_fields", {}
                    ),
                }

                if system_prompt:
                    body.update({"system": [{"text": system_prompt}]})

                streaming_response = self._stream_response(body=body)

                response = ""
                usage = None
                no_error = True

                async for text in streaming_response:
                    if request and await request.is_disconnected():
                        yield {"error": 400, "reason": "request aborted by user"}
                        await streaming_response.aclose()
                        break
                    elif text and isinstance(text, str):
                        response += text
                        yield text
                    elif isinstance(text, Dict) and "error" not in text:
                        usage = text
                    elif isinstance(text, Dict) and "error" in text:
                        no_error = False
                        yield text
                        await streaming_response.aclose()
                        break
                    else:
                        pass

                if no_error:
                    result = {
                        "response": response.strip(),
                        "metadata": {"usage": usage},
                    }

                    yield result

        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            yield {"error": 500, "reason": str(e)}

    async def _generate_response(self, body: Dict) -> Dict:
        try:
            response = await asyncio.to_thread(self.client.converse, **body)

            result = {
                "response": response.get("output", {})
                .get("message", {})
                .get("content", [{}])[0]
                .get("text")
                .strip(),
                "metadata": {"usage": response.get("usage", {})},
            }

            if response.get("metrics"):
                result["metadata"]["usage"].update(response.get("metrics"))

            return result

        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}

    async def _stream_response(self, body: Dict) -> AsyncGenerator:
        try:
            response = await asyncio.to_thread(self.client.converse_stream, **body)

            streaming_response = response.get("stream")

            stream_start = False

            for event in streaming_response:
                if event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                    if not stream_start:
                        yield (
                            event.get("contentBlockDelta")
                            .get("delta")
                            .get("text")
                            .strip()
                        )
                        stream_start = True
                    else:
                        yield event.get("contentBlockDelta").get("delta").get("text")
                elif event.get("metadata", {}).get("usage"):
                    usage = event.get("metadata").get("usage")

                    if event.get("metadata").get("metrics"):
                        usage.update(event.get("metadata").get("metrics"))

                    yield usage

        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            yield {"error": 500, "reason": str(e)}

    async def _chat_formatter(
        self,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
        do_json: Optional[bool] = False,
    ) -> List[Dict]:
        try:
            messages = []

            if chat_hist:
                for chat_log in chat_hist:
                    messages.append(
                        {
                            "role": chat_log.get("role"),
                            "content": [{"text": chat_log.get("content")}],
                        }
                    )

            if isinstance(user_message, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": user_message
                                + "\n(Respond in JSON and do not give any explanation or notes)"
                                if do_json
                                else user_message
                            }
                        ],
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

                if messages[-1].get("role") == "user":
                    messages[-1]["content"][0]["text"] = (
                        messages[-1]["content"][0]["text"]
                        + "\n(Respond in JSON and do not give any explanation or notes)"
                    )
                elif messages[-1].get("role") == "assistant":
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
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}
