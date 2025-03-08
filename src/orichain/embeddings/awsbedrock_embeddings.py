from typing import Any, List, Dict, Union
import traceback
import json

from botocore.exceptions import ClientError

import asyncio


class Embed(object):
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

        self.accept = {
            "amazon.titan-embed-text-v1": "application/json",
            "amazon.titan-embed-text-v2:0": "application/json",
            "cohere.embed-english-v3": "*/*",
            "cohere.embed-multilingual-v3": "*/*",
        }

        self.content_type = "application/json"

    async def __call__(
        self, text: Union[str, List[str]], model_name: str, **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        - Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            **kwargs: Additional keyword arguments for the embedding API

        - Returns:
            - Union[List[float], List[List[float]], Dict[str, Any]]:
            - Embeddings or error information
        """
        try:
            self.embedding_types = kwds.get("embedding_types")

            if isinstance(text, str):
                text = [text]

            # This can be removed, as all will be independent calls
            if "cohere" in model_name:
                if len(text) > 96:
                    return {
                        "error": 500,
                        "reason": "too many embedding requests being processed, lower the number",
                    }

            # creatings tasks
            tasks = []
            for sentence in text:
                if "cohere" in model_name:
                    if len(sentence) > 2048 and kwds.get("truncate", "NONE") == "NONE":
                        return {
                            "error": 500,
                            "reason": f"length of query is {len(sentence)}, please lower it to 2048",
                        }

                    body = {
                        "texts": [sentence],
                        "input_type": kwds.get("input_type", "search_query"),
                        "truncate": kwds.get("truncate", "NONE"),
                    }
                    if kwds.get("embedding_types"):
                        body.update({"embedding_types": [kwds.get("embedding_types")]})

                elif "amazon" in model_name:
                    if "v2" in model_name:
                        body = {
                            "inputText": sentence,
                            "dimensions": kwds.get(
                                "dimensions", 1024
                            ),  # Output dimensions can be: 256, 512 and 1024
                            "normalize": kwds.get("normalize", True),
                        }  # As recommended in docs for RAG
                    elif "v1" in model_name:
                        body = {"inputText": sentence}

                tasks.append(self._generate_embeddings(body=body, model_id=model_name))

            embeddings = await asyncio.gather(*tasks)

            for embedding in embeddings:
                if isinstance(embedding, Dict):
                    return embedding

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings

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

    async def _generate_embeddings(
        self, body: Dict, model_id: str
    ) -> Union[List[Union[float, int]], Dict]:
        try:
            response = await asyncio.to_thread(
                self.client.invoke_model,
                body=json.dumps(body),
                modelId=model_id,
                accept=self.accept.get(model_id),
                contentType=self.content_type,
            )

            response_body = json.loads(response.get("body").read())

            if "embeddings" in response_body:
                if self.embedding_types in response_body["embeddings"]:
                    return response_body["embeddings"].get(self.embedding_types)[0]
                else:
                    return response_body["embeddings"][0]
            else:
                return response_body["embedding"]
        except ClientError as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 400, "reason": e.response["Error"]["Message"]}
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
