from typing import Any, List, Union, Optional, Dict
import warnings
import traceback

from orichain.knowledge_base import pinecone_knowledgbase, chromadb_knowledgebase


class KnowledgeBase(object):
    default_knowledge_base = "pinecone"

    def __init__(self, vector_db_type: Optional[str], **kwds: Any) -> None:
        try:
            knowledge_base_handler = {
                "pinecone": pinecone_knowledgbase.DataBase,
                "chromadb": chromadb_knowledgebase.DataBase,
            }

            if not vector_db_type:
                warnings.warn(
                    f"Knowledge base type not defined hence defaulting to \
                    {self.default_knowledge_base}",
                    UserWarning,
                )
                self.vector_db_type = self.default_knowledge_base
            elif vector_db_type not in list(knowledge_base_handler.keys()):
                raise ValueError(
                    f"\nUnsupported knowledge base: {self.model_name}\nSupported knowledge bases are:"
                    f"\n- " + "\n- ".join(list(knowledge_base_handler.keys()))
                )
            else:
                self.vector_db_type = vector_db_type

            self.retriver = knowledge_base_handler.get(
                vector_db_type, self.default_knowledge_base
            )(**kwds)

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

    async def __call__(
        self,
        num_of_chunks: int,
        user_message_vector: Optional[List[Union[int, float]]] = None,
        **kwds: Any,
    ) -> Dict:
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

    async def fetch(
        self,
        ids: List[str],
        **kwds: Any,
    ) -> Dict:
        try:
            chunks = await self.retriver.fetch(
                ids=ids,
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
