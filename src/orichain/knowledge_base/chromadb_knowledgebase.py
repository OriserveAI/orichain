from typing import Any, List, Union, Dict
import traceback
import warnings
import asyncio


class DataBase(object):
    def __init__(self, **kwds) -> None:
        if not kwds.get("collection_name"):
            raise KeyError("Required `collection_name` not found")
        elif not kwds.get("embedding_function"):
            warnings.warn(
                "‘embedding_function’ has not been defined while initializing the KnowledgeBase class.\n"
                "You are using ChromaDB; if a collection has been created with an embedding model different from the "
                "default embedding model (`all-MiniLM-L6-v2`) that ChromaDB uses, YOU NEED TO PASS 'embedding_function' LIKE THIS:\n"
                "------------------------------------------------------\n"
                "class MyEmbeddingFunction(chromadb.EmbeddingFunction):\n"
                "\tdef __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:\n"
                "\t\tembeddings = []\n"
                "\t\tfor text in input:\n"
                "\t\t\tembeddings.append(embedding_model(text=text))\n"
                "\t\treturn embeddings",
                UserWarning,
            )
        else:
            pass

        __import__("pysqlite3")
        import sys

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

        import chromadb

        # TODO Need to change it from PersistentClient to HTTPClient mode, as PersistentClient is not prod friendly
        self.client = chromadb.PersistentClient(
            path=kwds.get("path", "/home/ubuntu/projects/chromadb")
        )
        self.collection = self.client.get_collection(
            name=kwds.get("collection_name"),
            embedding_function=kwds.get("embedding_function"),
        )

        self.collection.get()

    async def __call__(
        self,
        user_message_vector: List[Union[int, float]],
        num_of_chunks: int,
        **kwds: Any,
    ) -> Any:
        try:
            if kwds.get("collection_name"):
                if not kwds.get("embedding_function"):
                    warnings.warn(
                        "‘embedding_function’ has not been defined while calling the KnowledgeBase class (if you are changing"
                        "the `collection_name` while calling, then pass `embedding_function`"
                        "You are using ChromaDB; if a collection has been created with an embedding model different from the "
                        "default embedding model (`all-MiniLM-L6-v2`) that ChromaDB uses, YOU NEED TO PASS 'embedding_function' LIKE THIS:\n"
                        "------------------------------------------------------\n"
                        "class MyEmbeddingFunction(chromadb.EmbeddingFunction):\n"
                        "\tdef __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:\n"
                        "\t\tembeddings = []\n"
                        "\t\tfor text in input:\n"
                        "\t\t\tembeddings.append(embedding_model(text=text))\n"
                        "\t\treturn embeddings",
                        UserWarning,
                    )
                collection = self.client.get_collection(
                    name=kwds.get("collection_name"),
                    embedding_function=kwds.get("embedding_function"),
                )
            else:
                collection = self.collection

            chunks = await asyncio.to_thread(
                self._query,
                collection=collection,
                user_message_vector=user_message_vector,
                num_of_chunks=num_of_chunks,
                **kwds,
            )

            return chunks

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

    async def fetch(self, ids: List[str], **kwds: Any) -> Dict:
        try:
            if kwds.get("collection_name"):
                if not kwds.get("embedding_function"):
                    warnings.warn(
                        "‘embedding_function’ has not been defined while calling the KnowledgeBase class (if you are changing"
                        "the `collection_name` while calling, then pass `embedding_function`"
                        "You are using ChromaDB; if a collection has been created with an embedding model different from the "
                        "default embedding model (`all-MiniLM-L6-v2`) that ChromaDB uses, YOU NEED TO PASS 'embedding_function' LIKE THIS:\n"
                        "------------------------------------------------------\n"
                        "class MyEmbeddingFunction(chromadb.EmbeddingFunction):\n"
                        "\tdef __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:\n"
                        "\t\tembeddings = []\n"
                        "\t\tfor text in input:\n"
                        "\t\t\tembeddings.append(embedding_model(text=text))\n"
                        "\t\treturn embeddings",
                        UserWarning,
                    )

                collection = self.client.get_collection(
                    name=kwds.get("collection_name"),
                    embedding_function=kwds.get("embedding_function"),
                )
            else:
                collection = self.collection

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

    def _query(
        self,
        collection: Any,
        user_message_vector: List[Union[int, float]],
        num_of_chunks: int,
        **kwds: Any,
    ) -> Dict:
        result = collection.query(
            query_embeddings=user_message_vector,
            n_results=num_of_chunks,
            where=kwds.get("where"),
            where_document=kwds.get("where_document"),
            include=kwds.get("include")
            if kwds.get("include")
            else ["metadatas", "documents"],
        )

        return result
