from typing import Any, Optional, List, Dict, AsyncGenerator
import traceback
import warnings
import json

from fastapi import Request

from orichain.llm import (
    openai_llm,
    anthropicbedrock_llm,
    anthropic_llm,
    awsbedrock_llm,
    azureopenai_llm,
)


class LLM(object):
    default_model = "gpt-4o-mini"

    model_class = {
        "gpt-4o": "OpenAI",
        "gpt-4-turbo": "OpenAI",
        "gpt-4-turbo-preview": "OpenAI",
        "gpt-4o-mini": "OpenAI",
        "gpt-4": "OpenAI",
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

    def __init__(self, **kwds: Any) -> None:
        if not kwds.get("model_name"):
            warnings.warn(
                f"No 'model_name' specified, hence defaulting to {self.default_model} (OpenAI)",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)

        self.model_type = self.model_class.get(self.model_name)

        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"

        if not self.model_type:
            raise ValueError(
                f"\nUnsupported model: {self.model_name}\nSupported models are:"
                f"\n- " + "\n- ".join(list(self.model_class.keys()))
            )

        model_handler = {
            "OpenAI": openai_llm.Generate,
            "AWSBedrock": awsbedrock_llm.Generate,
            "AnthropicAWSBedrock": anthropicbedrock_llm.Generate,
            "Anthropic": anthropic_llm.Generate,
            "AzureOpenAI": azureopenai_llm.Generate,
        }

        self.model = model_handler.get(self.model_type)(**kwds)

    async def __call__(
        self,
        user_message: str,
        request: Optional[Request] = None,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: Optional[List[Dict[str, str]]] = None,
        sampling_paras: Optional[Dict] = {},
        extra_metadata: Optional[Dict] = {},
        do_json: Optional[bool] = False,
        **kwds: Any,
    ) -> Dict:
        try:
            if await self._model_n_model_type_validator(**kwds):
                model_name = kwds.pop("model_name", self.model_name)
            else:
                model_name = self.model_name

            if request and await request.is_disconnected():
                return {"error": 400, "reason": "request aborted by user"}
            else:
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

                if "error" not in result:
                    result.update(
                        {"message": user_message, "matched_sentence": matched_sentence}
                    )
                    if extra_metadata:
                        result["metadata"].update(extra_metadata)

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

    async def stream(
        self,
        user_message: str,
        request: Optional[Request] = None,
        matched_sentence: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        chat_hist: List = None,
        sampling_paras: Optional[Dict] = {},
        extra_metadata: Optional[Dict] = {},
        do_json: Optional[bool] = False,
        do_sse: Optional[bool] = True,
        **kwds: Any,
    ) -> AsyncGenerator:
        try:
            if await self._model_n_model_type_validator(**kwds):
                model_name = kwds.get("model_name", self.model_name)
            else:
                model_name = self.model_name

            if request and await request.is_disconnected():
                yield await self._format_sse(
                    {"error": 400, "reason": "request aborted by user"}, event="body"
                )
            else:
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
                                    "matched_sentence": matched_sentence,
                                }
                            )
                            if extra_metadata:
                                chunk["metadata"].update(extra_metadata)
                        if do_sse:
                            yield await self._format_sse(chunk, event="body")
                        else:
                            yield chunk

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
            yield await self._format_sse({"error": 500, "reason": str(e)}, event="body")

    async def _format_sse(self, data: Any, event=None) -> str:
        msg = f"data: {json.dumps(data)}\n\n"

        if event is not None:
            msg = f"event: {event}\n{msg}"

        return msg

    async def _model_n_model_type_validator(self, **kwds: Any) -> bool:
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
