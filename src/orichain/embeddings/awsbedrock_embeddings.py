from typing import Any, List, Dict, Union
import json

from concurrent.futures import ThreadPoolExecutor
import asyncio

from orichain import error_explainer


class Embed(object):
    """
    Synchronus AWS Bedrock Embeddings class to get embeddings for the given text(s).
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize AWS Bedrock client and set up API key.

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
        """

        from botocore.config import Config

        # Check for required parameters
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

        # Initialize AWS Bedrock client
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

        # Set up accept and content type for different models that wil be used by respective models
        self.accept = {
            "amazon.titan-embed-text-v1": "application/json",
            "amazon.titan-embed-text-v2:0": "application/json",
            "cohere.embed-english-v3": "*/*",
            "cohere.embed-multilingual-v3": "*/*",
        }
        self.content_type = "application/json"

    def __call__(
        self, text: Union[str, List[str]], model_name: str, **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            - input_type (str, optional): Type of input text. Default: "search_query"
            - truncate (str, optional): Truncate the input text. Default: "NONE"
            - dimensions (int, optional): Output dimensions. Default: 1024 (Output dimensions can be: 256, 512 and 1024)
            - normalize (bool, optional): Normalize the output. Default: True (As recommended in docs for RAG)
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
           (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
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

                tasks.append((body, model_name))

            # Use ThreadPoolExecutor for concurrent execution
            with ThreadPoolExecutor() as executor:
                embeddings = list(executor.map(self.worker, tasks))

            # check for errors
            for embedding in embeddings:
                if isinstance(embedding, Dict):
                    return embedding

            # return flattened list if only one embedding is present
            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def worker(self, task):
        """
        Worker function to execute in threads.

        Args:
            task (Tuple): Task to execute

        Returns:
            (Union[List[Union[float, int]], Dict]): Embeddings or error information
        """
        body, model_id = task
        return self._generate_embeddings(body=body, model_id=model_id)

    def _generate_embeddings(
        self, body: Dict, model_id: str
    ) -> Union[List[Union[float, int]], Dict]:
        """
        Generate embeddings for the given text.

        Args:
            body (Dict): Request body
            model_id (str): Model ID

        Returns:
            (Union[List[Union[float, int]], Dict]): Embeddings or error information
        """
        try:
            # Invoke model in a separate thread
            response = self.client.invoke_model(
                body=json.dumps(body),
                modelId=model_id,
                accept=self.accept.get(model_id),
                contentType=self.content_type,
            )

            # Extract embeddings from response
            response_body = json.loads(response.get("body").read())
            if "embeddings" in response_body:
                if self.embedding_types in response_body["embeddings"]:
                    return response_body["embeddings"].get(self.embedding_types)[0]
                else:
                    return response_body["embeddings"][0]
            else:
                return response_body["embedding"]

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncEmbed(object):
    """
    Asynchronus AWS Bedrock Embeddings class to get embeddings for the given text(s).
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Initialize AWS Bedrock client and set up API key.

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
        """

        from botocore.config import Config

        # Check for required parameters
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

        # Initialize AWS Bedrock client
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

        # Set up accept and content type for different models that wil be used by respective models
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

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - model_name (str): Name of the embedding model to use
            - input_type (str, optional): Type of input text. Default: "search_query"
            - truncate (str, optional): Truncate the input text. Default: "NONE"
            - dimensions (int, optional): Output dimensions. Default: 1024 (Output dimensions can be: 256, 512 and 1024)
            - normalize (bool, optional): Normalize the output. Default: True (As recommended in docs for RAG)
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
           (Union[List[float], List[List[float]], Dict[str, Any]]): Embeddings or error information
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

            # running tasks
            embeddings = await asyncio.gather(*tasks)

            # check for errors
            for embedding in embeddings:
                if isinstance(embedding, Dict):
                    return embedding

            # return flattened list if only one embedding is present
            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def _generate_embeddings(
        self, body: Dict, model_id: str
    ) -> Union[List[Union[float, int]], Dict]:
        """
        Generate embeddings for the given text.

        Args:
            body (Dict): Request body
            model_id (str): Model ID

        Returns:
            (Union[List[Union[float, int]], Dict]): Embeddings or error information
        """
        try:
            # Invoke model in a separate thread
            response = await asyncio.to_thread(
                self.client.invoke_model,
                body=json.dumps(body),
                modelId=model_id,
                accept=self.accept.get(model_id),
                contentType=self.content_type,
            )

            # Extract embeddings from response
            response_body = json.loads(response.get("body").read())
            if "embeddings" in response_body:
                if self.embedding_types in response_body["embeddings"]:
                    return response_body["embeddings"].get(self.embedding_types)[0]
                else:
                    return response_body["embeddings"][0]
            else:
                return response_body["embedding"]

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
