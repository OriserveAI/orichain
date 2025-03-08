from typing import Any, List, Dict, Union
import traceback


class Embed(object):
    def __init__(self, **kwds: Any) -> None:
        """
        Loads SentenceTransformer model and initializes it.

        Args:
            - model_name (str): name of hf model to be loaded
        """

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            install = (
                input(
                    "sentence-transformers is not installed. Do you want to install it now? (y/n): "
                )
                .strip()
                .lower()
            )
            if install == "y" or install == "yes":
                import subprocess

                subprocess.run(["pip", "install", "sentence-transformers"], check=True)
            else:
                raise ImportError(
                    f"sentence-transformers is required for embeddings functionalities ({kwds.get('model_name', 'NA')}). Please install it manually using `pip install orichain[sentence-transformers]' or 'pip install sentence-transformers==3.4.1`."
                )

        self.default_model_dir = kwds.get(
            "model_download_path", "/home/ubuntu/projects/models/embedding_models"
        )

        from sentence_transformers import SentenceTransformer
        import os

        if not os.path.isdir(f"{self.default_model_dir}/{kwds.get('model_name')}"):
            self.model = SentenceTransformer(
                model_name_or_path=kwds.get("model_name"),
                device=kwds.get("device", "cpu"),
                trust_remote_code=kwds.get("trust_remote_code", False),
                token=kwds.get("token", None),
            )
            self.model.save(f"{self.default_model_dir}/{kwds.get('model_name')}")
        else:
            self.model = SentenceTransformer(
                model_name_or_path=f"{self.default_model_dir}/{kwds.get('model_name')}",
                device=kwds.get("device", "cpu"),
                trust_remote_code=kwds.get("trust_remote_code", False),
                token=kwds.get("token", None),
                local_files_only=True,
            )

    async def __call__(
        self, text: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        - Args:
            - text (Union[str, List[str]]): Input text or list of texts
            **kwargs: Additional keyword arguments for the embedding API

        - Returns:
            - Union[List[float], List[List[float]], Dict[str, Any]]:
            - Embeddings or error information
        """
        try:
            if isinstance(text, str):
                text = [text]

            embeddings = self.model.encode(text)
            embeddings = embeddings.tolist()
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
