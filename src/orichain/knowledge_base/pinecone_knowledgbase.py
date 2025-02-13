from typing import Any, List, Union, Dict, Optional
import traceback
import warnings
import asyncio


class DataBase(object):
    def __init__(self, **kwds) -> None:
        if not kwds.get("api_key"):
            raise KeyError("Required `api_key` not found")
        elif not kwds.get("index_name"):
            raise KeyError("Required `index_name` not found")
        elif not kwds.get("namespace"):
            warnings.warn(
                """'namespace' has not been defined while initializing the KnowledgeBase class, \
                declare it while calling the it"""
            )

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
        try:
            if not kwds.get("namespace") and not self.namespace:
                raise KeyError("Required `namespace` not found")
            elif not user_message_vector and not kwds.get("id"):
                raise ValueError(
                    "Atleast one is required `user_message_vector` or `id`"
                )

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
            if not kwds.get("namespace") and not self.namespace:
                raise KeyError("Required `namespace` not found")

            chunks = await asyncio.to_thread(
                self.index.fetch,
                ids=ids,
                namespace=kwds.get("namespace") or self.namespace,
            )

            return chunks.to_dict()

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
