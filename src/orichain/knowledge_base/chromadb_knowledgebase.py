from typing import Any, List, Union, Dict

from orichain import error_explainer

import asyncio


class DataBase(object):
    """
    Synchronous way to interact with the Chromadb knowledge base
    """

    def __init__(self, **kwds) -> None:
        """Initializes the Chromadb client.

        Args:
            - path (str, optional): Path to the Chromadb Defualt: `/home/ubuntu/projects/chromadb`
            - collection_name (str): Chromadb collection name

        Raises:
            - KeyError: If required parameters are not found
        """
        if not kwds.get("collection_name"):
            raise KeyError("Required `collection_name` not found")

        ## WILL NOT BE USED as we are not passing text directly while retrieving the chunks in query and get methods, keep it here just for reference
        # elif not kwds.get("embedding_function"):
        #     warnings.warn(
        #         "‘embedding_function’ has not been defined while initializing the KnowledgeBase class.\n"
        #         "You are using ChromaDB; if a collection has been created with an embedding model different from the "
        #         "default embedding model (`all-MiniLM-L6-v2`) that ChromaDB uses, YOU NEED TO PASS 'embedding_function' LIKE THIS:\n"
        #         "------------------------------------------------------\n"
        #         "class MyEmbeddingFunction(chromadb.EmbeddingFunction):\n"
        #         "\tdef __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:\n"
        #         "\t\tembeddings = []\n"
        #         "\t\tfor text in input:\n"
        #         "\t\t\tembeddings.append(embedding_model(text=text))\n"
        #         "\t\treturn embeddings",
        #         UserWarning,
        #     )

        else:
            pass

        try:
            # Importing pysqlite3 as sqlite3
            __import__("pysqlite3")
            import sys

            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

            import chromadb

            # TODO Need to create a separate class for HTTPClient but now we are using PersistentClient

            # Initializing the Chromadb client
            self.client = chromadb.PersistentClient(
                path=kwds.get("path", "/home/ubuntu/projects/chromadb")
            )

            # Getting the collection
            self.collection = self.client.get_collection(
                name=kwds.get("collection_name")
            )
        except Exception as e:
            error_explainer(e)

    def __call__(
        self,
        user_message_vector: List[Union[int, float]],
        num_of_chunks: int,
        **kwds: Any,
    ) -> Any:
        """Retrieves the chunks from the knowledge base using query method

        Args:
            - user_message_vector (List[Union[int, float]]): Embedding of text
            - num_of_chunks (int): Number of chunks to retrieve

        Returns:
            Dict: Result of retrieving the chunks
        """
        try:
            # If collection_name is provided, get the collection
            if kwds.get("collection_name"):
                collection = self.client.get_collection(
                    name=kwds.get("collection_name")
                )
            else:
                collection = self.collection

            # Querying the collection
            chunks = collection.query(
                query_embeddings=user_message_vector,
                n_results=num_of_chunks,
                where=kwds.get("where"),
                where_document=kwds.get("where_document"),
                include=kwds.get("include")
                if kwds.get("include")
                else ["metadatas", "documents"],
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    def fetch(self, ids: List[str], **kwds: Any) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base using get method

        Args:
            - ids (List[str]): List of ids to fetch

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # If collection_name is provided, get the collection
            if kwds.get("collection_name"):
                collection = self.client.get_collection(
                    name=kwds.get("collection_name")
                )
            else:
                collection = self.collection

            # Fetching the chunks based on the ids
            chunks = collection.get(
                ids=ids,
                limit=kwds.get("limit"),
                offset=kwds.get("offset"),
                where=kwds.get("where"),
                where_document=kwds.get("where_document"),
                include=kwds.get("include")
                if kwds.get("include")
                else ["metadatas", "documents"],
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncDataBase(object):
    """
    Asynchronous way to interact with the Chromadb knowledge base
    """

    def __init__(self, **kwds) -> None:
        """Initializes the Chromadb client.

        Args:
            - path (str): Path to the Chromadb
            - collection_name (str): Chromadb collection name

        Raises:
            - KeyError: If required parameters are not found
        """
        if not kwds.get("collection_name"):
            raise KeyError("Required `collection_name` not found")

        ## WILL NOT BE USED as we are not passing text directly while retrieving the chunks in query and get methods, keep it here just for reference
        # elif not kwds.get("embedding_function"):
        #     warnings.warn(
        #         "‘embedding_function’ has not been defined while initializing the KnowledgeBase class.\n"
        #         "You are using ChromaDB; if a collection has been created with an embedding model different from the "
        #         "default embedding model (`all-MiniLM-L6-v2`) that ChromaDB uses, YOU NEED TO PASS 'embedding_function' LIKE THIS:\n"
        #         "------------------------------------------------------\n"
        #         "class MyEmbeddingFunction(chromadb.EmbeddingFunction):\n"
        #         "\tdef __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:\n"
        #         "\t\tembeddings = []\n"
        #         "\t\tfor text in input:\n"
        #         "\t\t\tembeddings.append(embedding_model(text=text))\n"
        #         "\t\treturn embeddings",
        #         UserWarning,
        #     )

        else:
            pass

        try:
            # Importing pysqlite3 as sqlite3
            __import__("pysqlite3")
            import sys

            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

            import chromadb

            # TODO Need to create a separate class for HTTPClient but now we are using PersistentClient

            # Initializing the Chromadb client
            self.client = chromadb.PersistentClient(
                path=kwds.get("path", "/home/ubuntu/projects/chromadb")
            )

            # Getting the collection
            self.collection = self.client.get_collection(
                name=kwds.get("collection_name")
            )
        except Exception as e:
            error_explainer(e)

    async def __call__(
        self,
        user_message_vector: List[Union[int, float]],
        num_of_chunks: int,
        **kwds: Any,
    ) -> Any:
        """Retrieves the chunks from the knowledge base using query method

        Args:
            - user_message_vector (List[Union[int, float]]): Embedding of text
            - num_of_chunks (int): Number of chunks to retrieve

        Returns:
            Dict: Result of retrieving the chunks
        """
        try:
            # If collection_name is provided, get the collection
            if kwds.get("collection_name"):
                collection = self.client.get_collection(
                    name=kwds.get("collection_name")
                )
            else:
                collection = self.collection

            # Querying the collection
            chunks = await asyncio.to_thread(
                collection.query,
                query_embeddings=user_message_vector,
                n_results=num_of_chunks,
                where=kwds.get("where"),
                where_document=kwds.get("where_document"),
                include=kwds.get("include")
                if kwds.get("include")
                else ["metadatas", "documents"],
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}

    async def fetch(self, ids: List[str], **kwds: Any) -> Dict:
        """Fetches the chunks based on the ids from the knowledge base using get method

        Args:
            - ids (List[str]): List of ids to fetch

        Returns:
            Dict: Result of fetching the chunks
        """
        try:
            # If collection_name is provided, get the collection
            if kwds.get("collection_name"):
                collection = self.client.get_collection(
                    name=kwds.get("collection_name")
                )
            else:
                collection = self.collection

            # Fetching the chunks based on the ids
            chunks = await asyncio.to_thread(
                collection.get,
                ids=ids,
                limit=kwds.get("limit"),
                offset=kwds.get("offset"),
                where=kwds.get("where"),
                where_document=kwds.get("where_document"),
                include=kwds.get("include")
                if kwds.get("include")
                else ["metadatas", "documents"],
            )

            return chunks

        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
