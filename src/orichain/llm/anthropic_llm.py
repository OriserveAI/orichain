from typing import Any, List, Dict, Optional, AsyncGenerator, Union
import traceback
from fastapi import Request


class Generate(object):
    def __init__(self, **kwds: Any) -> None:
        """
        Initialize Anthropic client and set up API key.

        Args:
            - api_key (str): api key
        """
        from httpx import Timeout

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

        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=kwds.get("api_key"),
            timeout=kwds.get("timeout")
            or Timeout(60.0, read=5.0, write=10.0, connect=2.0),
            max_retries=kwds.get("max_retries") or 2,
        )

    async def __call__(
        self,
        model_name: str,
        user_message: Union[str, List[Dict[str, str]]],
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Dict:
        try:
            messages = await self._chat_formatter(
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            if isinstance(messages, Dict):
                return messages

            if "max_tokens" not in sampling_paras:
                sampling_paras["max_tokens"] = 512

            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            message = await self.client.with_options(
                timeout=kwds.get("timeout")
            ).messages.create(
                system=system_prompt if system_prompt else None,
                messages=messages,
                model=model_name,
                **sampling_paras,
            )

            result = {
                "response": "{" + message.content[0].text
                if do_json
                else message.content[0].text,
                "metadata": {"usage": message.usage.to_dict()},
            }

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
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> AsyncGenerator:
        try:
            messages = await self._chat_formatter(
                user_message=user_message, chat_hist=chat_hist, do_json=do_json
            )

            if isinstance(messages, Dict):
                yield messages
            else:
                if "max_tokens" not in sampling_paras:
                    sampling_paras["max_tokens"] = 512

                async with self.client.messages.stream(
                    system=system_prompt,
                    messages=messages,
                    model=model_name,
                    **sampling_paras,
                ) as stream:
                    if do_json:
                        yield "{"
                    async for chunk in stream.text_stream:
                        if request and await request.is_disconnected():
                            yield {"error": 400, "reason": "request aborted by user"}
                            await stream.close()
                            break

                        if chunk:
                            yield chunk

                final_response = await stream.get_final_message()

                result = {
                    "response": "{" + final_response.content[0].text
                    if do_json
                    else final_response.content[0].text,
                    "metadata": {"usage": final_response.usage.to_dict()},
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

    async def _chat_formatter(
        self,
        user_message: Union[str, List[Dict[str, str]]],
        chat_hist: Optional[List[Dict[str, str]]] = None,
        do_json: Optional[bool] = False,
    ) -> List[Dict]:
        try:
            messages = []

            if chat_hist:
                messages.extend(chat_hist)

            if isinstance(user_message, str):
                messages.append({"role": "user", "content": user_message})
            elif isinstance(user_message, List):
                messages.extend(user_message)
            else:
                raise KeyError(
                    """invalid user messages format, this error should not happen due to the validation steps being used.\
                    Please check the validation function"""
                )

            if do_json:
                messages.append({"role": "assistant", "content": "{"})

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
