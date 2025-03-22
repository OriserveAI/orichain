from typing import Any, List, Union, Optional, Dict
import warnings

from orichain.knowledge_base import pinecone_knowledgbase, chromadb_knowledgebase
from orichain import error_explainer

DEFAULT_KNOWLEDGE_BASE = "pinecone"


class KnowledgeBase(object):
    """
    Synchronous way to retrieve chunks from the knowledge base
    """

    default_knowledge_base = DEFAULT_KNOWLEDGE_BASE

    def __init__(self, vector_db_type: Optional[str], **kwds: Any) -> None:
        """Initializes the knowledge base.

        Args:
            vector_db_type (str, optional): Type of knowledge base. Default: pinecone

        Authentication parameters by provider:

            Pinecone:
            api_key (str): Pinecone API key
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace

            ChromaDB:
            collection_name (str): ChromaDB collection name
            path (str, optional): Path to the ChromaDB database Default: `/home/ubuntu/projects/chromadb`

        Raises:
            ValueError: If the knowledge base type is not supported
            KeyError: If the required params is not found

        Warns:
            UserWarning: If the knowledge base type is not defined Default: pinecone
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
            num_of_chunks (int): Number of chunks to retrieve
            user_message_vector (Optional[List[Union[int, float]]]): Embedding of text. Defaults to None.

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            ValueError: If `user_message_vector` is needed except for pinecone but if ids are also not provided for pinecone this error will be raised
            KeyError: If required `namespace` is not found for pinecone
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
            ids (List[str]): List of ids to fetch

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
    Asynchronous way to retrieve chunks from the knowledge base
    """

    default_knowledge_base = DEFAULT_KNOWLEDGE_BASE

    def __init__(self, vector_db_type: Optional[str], **kwds: Any) -> None:
        """Initializes the knowledge base.

        Args:
            vector_db_type (str, optional): Type of knowledge base. Default: pinecone

        Authentication parameters by provider:

            Pinecone:
            api_key (str): Pinecone API key
            index_name (str): Pinecone index name
            namespace (str): Pinecone namespace

            ChromaDB:
            collection_name (str): ChromaDB collection name
            path (str, optional): Path to the ChromaDB database Default: `/home/ubuntu/projects/chromadb`

        Raises:
            ValueError: If the knowledge base type is not supported
            KeyError: If the required params is not found

        Warns:
            UserWarning: If the knowledge base type is not defined Default: pinecone
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
            num_of_chunks (int): Number of chunks to retrieve
            user_message_vector (Optional[List[Union[int, float]]]): Embedding of text. Defaults to None.

        Returns:
            Dict: Result of retrieving the chunks

        Raises:
            ValueError: If `user_message_vector` is needed except for pinecone but if ids are also not provided for pinecone this error will be raised
            KeyError: If required `namespace` is not found for pinecone
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
            ids (List[str]): List of ids to fetch

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
