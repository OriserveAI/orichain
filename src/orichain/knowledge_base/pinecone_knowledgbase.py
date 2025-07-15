from typing import Any, List, Union, Dict, Optional
import asyncio

from orichain import error_explainer


class DataBase(object):
    """
    Synchronous way to interact with the Pinecone knowledge base
    """

    def __init__(self, **kwds) -> None:
        """Initializes the Pinecone GRPC client.

        Args:
            - api_key (str): Pinecone API key
            - index_name (str): Pinecone index name
            - namespace (str): Pinecone namespace

        Raises:
            - KeyError: If required parameters are not found
        """
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("index_name"):
            raise KeyError("Required `index_name` not found")
        elif not kwds.get("namespace"):
            raise KeyError("Required `namespace` not found")

        from pinecone.grpc import PineconeGRPC

        self.namespace = kwds.get("namespace")
        client = PineconeGRPC(api_key=kwds.get("api_key"))
        self.index = client.Index(kwds.get("index_name"))

    def __call__(
        self,
        num_of_chunks: int,
        user_message_vector: Optional[List[Union[int, float]]] = None,
        **kwds: Any,
    ) -> Dict:
        """Retrieves chunks from the knowledge base using query method

        Args:
            - vector (List[float]): The query vector. This should be the same length as the dimension of the index
              being queried. Each `query()` request can contain only one of the parameters `id` or `vector`.. [optional]
            - id (str): The unique ID of the vector to be used as a query vector.
              Each `query()` request can contain only one of the parameters `vector` or  `id`.. [optional]
            - top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            - namespace (str): The namespace to fetch vectors from.
              If not specified, the default namespace is used. [optional]
            - filter (Dict[str, Union[str, float, int, bool, List, dict]]):
              The filter to apply. You can use vector metadata to limit your search.
              See https://www.pinecone.io/docs/metadata-filtering/ [optional]
            - include_values (bool): Indicates whether vector values are included in the response.
              If omitted the server will use the default value of False [optional]
            - include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
              If omitted the server will use the default value of False  [optional]
            - sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
              Expected to be either a SparseValues object or a dict of the form:
              {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            - ValueError: If `user_message_vector` or `id` is not provided
        """
        try:
            if not user_message_vector and not kwds.get("id"):
                raise ValueError(
                    "Atleast one is required `user_message_vector` or `id`"
                )

            # Querying the chunks from the knowledge base
            chunks = self.index.query(
                vector=user_message_vector,
                top_k=num_of_chunks,
                id=kwds.get("id"),
                sparse_vector=kwds.get("sparse_vector"),
                include_values=kwds.get("include_values"),
                filter=kwds.get("filter"),
                include_metadata=kwds.get("include_metadata", True),
                namespace=kwds.get("namespace") or self.namespace,
            )

            return chunks.to_dict()

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def fetch(self, ids: List[str], **kwds: Any) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base using fetch method

        Args:
            - ids (List[str]): List of ids to fetch
            - namespace (str, optional): The namespace to fetch vectors from. If not specified, the default namespace is used.

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # Fetching the chunks based on the ids
            chunks = self.index.fetch(
                ids=ids,
                namespace=kwds.get("namespace") or self.namespace,
            )

            return chunks.to_dict()

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncDataBase(object):
    """
    Asynchronous way to interact with the Pinecone knowledge base
    """

    def __init__(self, **kwds) -> None:
        """Initializes the Pinecone GRPC client.

        Args:
            - api_key (str): Pinecone API key
            - index_name (str): Pinecone index name
            - namespace (str): Pinecone namespace

        Raises:
            - KeyError: If required parameters are not found
        """
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("index_name"):
            raise KeyError("Required `index_name` not found")
        elif not kwds.get("namespace"):
            raise KeyError("Required `namespace` not found")

        from pinecone.grpc import PineconeGRPC

        self.namespace = kwds.get("namespace")
        client = PineconeGRPC(api_key=kwds.get("api_key"))
        self.index = client.Index(kwds.get("index_name"))

    async def __call__(
        self,
        num_of_chunks: int,
        user_message_vector: Optional[List[Union[int, float]]] = None,
        **kwds: Any,
    ) -> Dict:
        """Retrieves chunks from the knowledge base using query method

        Args:
            - vector (List[float]): The query vector. This should be the same length as the dimension of the index
              being queried. Each `query()` request can contain only one of the parameters `id` or `vector`.. [optional]
            - id (str): The unique ID of the vector to be used as a query vector.
              Each `query()` request can contain only one of the parameters `vector` or  `id`.. [optional]
            - top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            - namespace (str): The namespace to fetch vectors from.
              If not specified, the default namespace is used. [optional]
            - filter (Dict[str, Union[str, float, int, bool, List, dict]]):
              The filter to apply. You can use vector metadata to limit your search.
              See https://www.pinecone.io/docs/metadata-filtering/ [optional]
            - include_values (bool): Indicates whether vector values are included in the response.
              If omitted the server will use the default value of False [optional]
            - include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
              If omitted the server will use the default value of False  [optional]
            - sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
              Expected to be either a SparseValues object or a dict of the form:
              {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            - ValueError: If `user_message_vector` or `id` is not provided
        """
        try:
            if not user_message_vector and not kwds.get("id"):
                raise ValueError(
                    "Atleast one is required `user_message_vector` or `id`"
                )

            # Querying the chunks from the knowledge base
            chunks = await asyncio.to_thread(
                self.index.query,
                vector=user_message_vector,
                top_k=num_of_chunks,
                id=kwds.get("id"),
                sparse_vector=kwds.get("sparse_vector"),
                include_values=kwds.get("include_values"),
                filter=kwds.get("filter"),
                include_metadata=kwds.get("include_metadata", True),
                namespace=kwds.get("namespace") or self.namespace,
            )

            return chunks.to_dict()

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def fetch(self, ids: List[str], **kwds: Any) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base using fetch method

        Args:
            - ids (List[str]): List of ids to fetch
            - namespace (str, optional): The namespace to fetch vectors from. If not specified, the default namespace is used.

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # Fetching the chunks based on the ids
            chunks = await asyncio.to_thread(
                self.index.fetch,
                ids=ids,
                namespace=kwds.get("namespace") or self.namespace,
            )

            return chunks.to_dict()

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
