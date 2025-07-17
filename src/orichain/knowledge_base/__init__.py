from typing import Any, List, Union, Optional, Dict
import warnings

from orichain.knowledge_base import pinecone_knowledgbase, chromadb_knowledgebase
from orichain import error_explainer

DEFAULT_KNOWLEDGE_BASE = "pinecone"


class KnowledgeBase(object):
    """
    Synchronous interface for interacting with vector databases.

    This class provides a unified API to communicate with supported vector databases.
    Currently, Pinecone and ChromaDB are supported.
    """

    default_knowledge_base = DEFAULT_KNOWLEDGE_BASE

    def __init__(self, vector_db_type: Optional[str], **kwds: Any) -> None:
        """Initializes the knowledge base.

        Args:
            - vector_db_type (str, optional): Type of knowledge base. Default: pinecone

        **Authentication parameters by provider:**

            **Pinecone:**
                - api_key (str): Pinecone API key
                - index_name (str): Pinecone index name
                - namespace (str): Pinecone namespace

            **ChromaDB:**
                - collection_name (str): ChromaDB collection name
                - path (str, optional): Path to the ChromaDB database Default: `/home/ubuntu/projects/chromadb`

        Raises:
            - ValueError: If the knowledge base type is not supported
            - KeyError: If the required params is not found

        Warns:
            - UserWarning: If the knowledge base type is not defined Default: pinecone
        """
        try:
            # Dictionary mapping vector database types to their respective handler classes
            knowledge_base_handler = {
                "pinecone": pinecone_knowledgbase.DataBase,
                "chromadb": chromadb_knowledgebase.DataBase,
            }

            # If no vector_db_type is provided, default to pinecone
            if not vector_db_type:
                warnings.warn(
                    f"\nKnowledge base type not defined hence defaulting to \
                    {self.default_knowledge_base}",
                    UserWarning,
                )
                self.vector_db_type = self.default_knowledge_base
            # If vector_db_type is not supported, raise a ValueError
            elif vector_db_type not in list(knowledge_base_handler.keys()):
                raise ValueError(
                    f"\nUnsupported knowledge base: {self.model_name}\nSupported knowledge bases are:"
                    f"\n- " + "\n- ".join(list(knowledge_base_handler.keys()))
                )
            else:
                self.vector_db_type = vector_db_type

            # Initialize the knowledge base handler
            self.retriver = knowledge_base_handler.get(
                vector_db_type, self.default_knowledge_base
            )(**kwds)

        except Exception as e:
            error_explainer(e)

    def __call__(
        self,
        num_of_chunks: int,
        user_message_vector: Optional[List[Union[int, float]]] = None,
        **kwds: Any,
    ) -> Dict:
        """Retrieves the chunks from the knowledge base

        Args:
            - user_message_vector (Optional[List[Union[int, float]]]): Embedding of the text. Defaults to None.
            - num_of_chunks (int): Number of chunks to retrieve

            **Retrieval Arguments by VectorDB:**

                **Pinecone:**
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
                      ``{'indices': List[int], 'values': List[float]}``, where the lists each have the same length.

                **ChromaDB:**
                    - collection_name (str, optional): The name of the collection to get documents from. Defaults to the collection_name set during class instantiation.
                    - where (Dict, optional):  A Where type dict used to filter results by. E.g. ``{$and: [{"color" : "red"}, {"price": 4.20}]}``. Default: None.
                    - where_document (Dict, optional): A WhereDocument type dict used to filter by the documents. E.g. ``{"$contains" : "hello"}``. Default: None.
                    - include (List, optional): A list of what to include in the results.
                      Can contain ``"embeddings"``, ``"metadatas"``, ``"documents"``, ``"distances"``.
                      Ids are always included. Defaults to ``["metadatas", "documents", "distances"]``.
                      Default: ``["metadatas", "documents"]``

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            - ValueError: If `user_message_vector` is needed except for pinecone but if ids are also not provided for pinecone this error will be raised
            - KeyError: If required `namespace` is not found for pinecone
        """
        try:
            if not user_message_vector and not self.vector_db_type == "pinecone":
                raise ValueError("`user_message_vector` is needed except for pinecone")

            chunks = self.retriver(
                user_message_vector=user_message_vector,
                num_of_chunks=num_of_chunks,
                **kwds,
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def fetch(
        self,
        ids: List[str],
        **kwds: Any,
    ) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base

        Args:
            - ids (List[str]): List of ids to fetch

            **Retrieval Arguments by VectorDB:**

                **Pinecone:**
                    - namespace (str, optional): The namespace to fetch vectors from. If not specified, the default namespace is used.

                **ChromaDB:**
                    - collection_name (str, optional): The name of the collection to fetch documents from. Defaults to the collection_name set during class instantiation.
                    - limit (int, optional): The number of documents to return. Default: None.
                    - offset (int, optional): The offset to start returning results from. Useful for paging results with limit. Default: None.
                    - where (Dict, optional):  A Where type dict used to filter results by. E.g. ``{$and: [{"color" : "red"}, {"price": 4.20}]}``. Default: None.
                    - where_document (Dict, optional): A WhereDocument type dict used to filter by the documents. E.g. ``{"$contains" : "hello"}``. Default: None.
                    - include (List, optional): A list of what to include in the results.
                      Can contain ``"embeddings"``, ``"metadatas"``, ``"documents"``, ``"distances"``.
                      Ids are always included. Defaults to ``["metadatas", "documents", "distances"]``.
                      Default: ``["metadatas", "documents"]``

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # Fetching the chunks based on the ids
            chunks = self.retriver.fetch(
                ids=ids,
                **kwds,
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncKnowledgeBase(object):
    """
    Asynchronous interface for interacting with vector databases.

    This class provides a unified API to communicate with supported vector databases.
    Currently, Pinecone and ChromaDB are supported.
    """

    default_knowledge_base = DEFAULT_KNOWLEDGE_BASE

    def __init__(self, vector_db_type: Optional[str], **kwds: Any) -> None:
        """Initializes the knowledge base.

        Args:
            - vector_db_type (str, optional): Type of knowledge base. Default: pinecone

        **Authentication parameters by provider:**

            **Pinecone:**
                - api_key (str): Pinecone API key
                - index_name (str): Pinecone index name
                - namespace (str): Pinecone namespace

            **ChromaDB:**
                - collection_name (str): ChromaDB collection name
                - path (str, optional): Path to the ChromaDB database Default: `/home/ubuntu/projects/chromadb`

        Raises:
            - ValueError: If the knowledge base type is not supported
            - KeyError: If the required params is not found

        Warns:
            - UserWarning: If the knowledge base type is not defined Default: pinecone
        """
        try:
            # Dictionary mapping vector database types to their respective handler classes
            knowledge_base_handler = {
                "pinecone": pinecone_knowledgbase.AsyncDataBase,
                "chromadb": chromadb_knowledgebase.AsyncDataBase,
            }

            # If no vector_db_type is provided, default to pinecone
            if not vector_db_type:
                warnings.warn(
                    f"\nKnowledge base type not defined hence defaulting to \
                    {self.default_knowledge_base}",
                    UserWarning,
                )
                self.vector_db_type = self.default_knowledge_base
            # If vector_db_type is not supported, raise a ValueError
            elif vector_db_type not in list(knowledge_base_handler.keys()):
                raise ValueError(
                    f"\nUnsupported knowledge base: {self.model_name}\nSupported knowledge bases are:"
                    f"\n- " + "\n- ".join(list(knowledge_base_handler.keys()))
                )
            else:
                self.vector_db_type = vector_db_type

            # Initialize the knowledge base handler
            self.retriver = knowledge_base_handler.get(
                vector_db_type, self.default_knowledge_base
            )(**kwds)

        except Exception as e:
            error_explainer(e)

    async def __call__(
        self,
        num_of_chunks: int,
        user_message_vector: Optional[List[Union[int, float]]] = None,
        **kwds: Any,
    ) -> Dict:
        """Retrieves the chunks from the knowledge base

        Args:
            - num_of_chunks (int): Number of chunks to retrieve
            - user_message_vector (Optional[List[Union[int, float]]]): Embedding of text. Defaults to None.

            **Retrieval Arguments by VectorDB:**

                **Pinecone:**
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
                      ``{'indices': List[int], 'values': List[float]}``, where the lists each have the same length.

                **ChromaDB:**
                    - collection_name (str, optional): The name of the collection to get documents from. Defaults to the collection_name set during class instantiation.
                    - where (Dict, optional):  A Where type dict used to filter results by. E.g. ``{$and: [{"color" : "red"}, {"price": 4.20}]}``. Default: None.
                    - where_document (Dict, optional): A WhereDocument type dict used to filter by the documents. E.g. ``{"$contains" : "hello"}``. Default: None.
                    - include (List, optional): A list of what to include in the results.
                      Can contain ``"embeddings"``, ``"metadatas"``, ``"documents"``, ``"distances"``.
                      Ids are always included. Defaults to ``["metadatas", "documents", "distances"]``.
                      Default: ``["metadatas", "documents"]``

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            - ValueError: If `user_message_vector` is needed except for pinecone but if ids are also not provided for pinecone this error will be raised
            - KeyError: If required `namespace` is not found for pinecone
        """
        try:
            if not user_message_vector and not self.vector_db_type == "pinecone":
                raise ValueError("`user_message_vector` is needed except for pinecone")

            chunks = await self.retriver(
                user_message_vector=user_message_vector,
                num_of_chunks=num_of_chunks,
                **kwds,
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def fetch(
        self,
        ids: List[str],
        **kwds: Any,
    ) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base

        Args:
            - ids (List[str]): List of ids to fetch

            **Retrieval Arguments by VectorDB:**

                **Pinecone:**
                    - namespace (str, optional): The namespace to fetch vectors from. If not specified, the default namespace is used.

                **ChromaDB:**
                    - collection_name (str, optional): The name of the collection to fetch documents from. Defaults to the collection_name set during class instantiation.
                    - limit (int, optional): The number of documents to return. Default: None.
                    - offset (int, optional): The offset to start returning results from. Useful for paging results with limit. Default: None.
                    - where (Dict, optional):  A Where type dict used to filter results by. E.g. ``{$and: [{"color" : "red"}, {"price": 4.20}]}``. Default: None.
                    - where_document (Dict, optional): A WhereDocument type dict used to filter by the documents. E.g. ``{"$contains" : "hello"}``. Default: None.
                    - include (List, optional): A list of what to include in the results.
                      Can contain ``"embeddings"``, ``"metadatas"``, ``"documents"``, ``"distances"``.
                      Ids are always included. Defaults to ``["metadatas", "documents", "distances"]``.
                      Default: ``["metadatas", "documents"]``

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # Fetching the chunks based on the ids
            chunks = await self.retriver.fetch(
                ids=ids,
                **kwds,
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
