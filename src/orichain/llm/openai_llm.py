from typing import Dict, List, Optional, AsyncGenerator
import traceback
import asyncio

from fastapi import Request


class Generate(object):
    def __init__(self, **kwds) -> None:
        """
        Initialize OpenAI client and set up API key.

        Args:
            - api_key (str): OpenAI API key
        """
        from httpx import Timeout

        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
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

        from openai import AsyncOpenAI
        import tiktoken

        self.client = AsyncOpenAI(
            api_key=kwds.get("api_key"),
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
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> Dict:
        try:
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            if isinstance(messages, Dict):
                return messages

            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}

            completion = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **sampling_paras,
                response_format={"type": "json_object"}
                if do_json
                else {"type": "text"},
            )

            result = {
                "response": completion.choices[0].message.content,
                "metadata": {"usage": completion.usage.to_dict()},
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
        user_message: str,
        request: Optional[Request] = None,
        chat_hist: Optional[List[str]] = None,
        sampling_paras: Optional[Dict] = {},
        system_prompt: Optional[str] = None,
        do_json: Optional[bool] = False,
    ) -> AsyncGenerator:
        try:
            messages = await self._chat_formatter(
                user_message=user_message,
                chat_hist=chat_hist,
                system_prompt=system_prompt,
            )

            if isinstance(messages, Dict):
                yield messages

            else:
                completion = await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    **sampling_paras,
                    response_format={"type": "json_object"}
                    if do_json
                    else {"type": "text"},
                )

                response = ""

                async for chunk in completion:
                    if request and await request.is_disconnected():
                        yield {"error": 400, "reason": "request aborted by user"}
                        await completion.close()
                        break
                    else:
                        if chunk.choices[0].delta.content:
                            response += chunk.choices[0].delta.content
                            yield chunk.choices[0].delta.content

                prompt_tokens, completion_tokens = await asyncio.gather(
                    self.num_tokens_from_string(
                        string=system_prompt, model_name=model_name
                    ),
                    self.num_tokens_from_string(string=response, model_name=model_name),
                )

                result = {
                    "response": response,
                    "metadata": {
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                    },
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
        user_message: str,
        chat_hist: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if chat_hist:
                messages.extend(chat_hist)

            messages.append({"role": "user", "content": user_message})

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

    async def num_tokens_from_string(
        self, model_name: str = "gpt-3.5-turbo", string: str = None
    ) -> int:
        """Returns the number of tokens in a text string."""
        if string and model_name:
            encoding = self.tiktoken.encoding_for_model(model_name)
            num_tokens = len(encoding.encode(string))
            return num_tokens
        else:
            return 0
