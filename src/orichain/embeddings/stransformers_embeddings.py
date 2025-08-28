from typing import Any, List, Dict, Union
from orichain import error_explainer

import asyncio

VERSION = "3.4.1"


class Embed(object):
    """
    Synchronous Embed class to get embeddings from SentenceTransformers.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Loads SentenceTransformer model and initializes it.

        Args:
            - model_name (str): name of hf model to be loaded
            - model_download_path (str): path to download the model
            - device (str): device to be used for inference
            - trust_remote_code (bool): whether to trust remote code
            - token (str): token for downloading model

        Raises:
            - ImportError: If sentence-transformers is not installed, it raises ImportError.
        """

        # Check if sentence-transformers is installed
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

                subprocess.run(
                    ["pip", "install", f"sentence-transformers=={VERSION}"], check=True
                )
            else:
                raise ImportError(
                    f"sentence-transformers is required for embeddings functionalities ({kwds.get('model_name', 'NA')}). Please install it manually using `pip install orichain[sentence-transformers]' or 'pip install sentence-transformers=={VERSION}`."
                )

        self.default_model_dir = kwds.get(
            "model_download_path", "/home/ubuntu/projects/models/embedding_models"
        )

        from sentence_transformers import SentenceTransformer
        import os

        # Check if model is already downloaded
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

    def __call__(
        self, text: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        """
        Get embeddings for the given text(s).

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
            Union[List[float], List[List[float]], np.ndarray, torch.Tensor, Dict[str, Any]]: Embeddings in the requested format, or error information.
        """
        try:
            if isinstance(text, str):
                text = [text]

            embeddings = self.model.encode(
                sentences=text,
                prompt_name=kwds.get("prompt_name", None),
                prompt=kwds.get("prompt", None),
                output_value=kwds.get("output_value", "sentence_embedding"),
                show_progress_bar=kwds.get("show_progress_bar", False),
                precision=kwds.get("precision", "float32"),
                batch_size=kwds.get("batch_size", 32),
                convert_to_tensor=kwds.get("convert_to_tensor", False),
                convert_to_numpy=kwds.get("convert_to_numpy", False),
                device=kwds.get("device", None),
                normalize_embeddings=kwds.get("normalize_embeddings", False),
            )

            if not kwds.get("convert_to_tensor") and not kwds.get("convert_to_numpy"):
                embeddings = [embedding.tolist() for embedding in embeddings]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}


class AsyncEmbed(object):
    """
    Asynchronous Embed class to get embeddings from SentenceTransformers.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Loads SentenceTransformer model and initializes it.

        Args:
            - model_name (str): name of hf model to be loaded
            - model_download_path (str): path to download the model
            - device (str): device to be used for inference
            - trust_remote_code (bool): whether to trust remote code
            - token (str): token for downloading model

        Raises:
            - ImportError: If sentence-transformers is not installed, it raises ImportError.
        """

        # Check if sentence-transformers is installed
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

                subprocess.run(
                    ["pip", "install", f"sentence-transformers=={VERSION}"], check=True
                )
            else:
                raise ImportError(
                    f"sentence-transformers is required for embeddings functionalities ({kwds.get('model_name', 'NA')}). Please install it manually using `pip install orichain[sentence-transformers]' or 'pip install sentence-transformers=={VERSION}`."
                )

        self.default_model_dir = kwds.get(
            "model_download_path", "/home/ubuntu/projects/models/embedding_models"
        )

        from sentence_transformers import SentenceTransformer
        import os

        # Check if model is already downloaded
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

        Args:
            - text (Union[str, List[str]]): Input text or list of texts
            - **kwargs: Additional keyword arguments for the embedding API

        Returns:
            Union[List[float], List[List[float]], np.ndarray, torch.Tensor, Dict[str, Any]]: Embeddings in the requested format, or error information.
        """
        try:
            if isinstance(text, str):
                text = [text]

            embeddings = await asyncio.to_thread(
                self.model.encode,
                sentences=text,
                prompt_name=kwds.get("prompt_name", None),
                prompt=kwds.get("prompt", None),
                output_value=kwds.get("output_value", "sentence_embedding"),
                show_progress_bar=kwds.get("show_progress_bar", False),
                precision=kwds.get("precision", "float32"),
                batch_size=kwds.get("batch_size", 32),
                convert_to_tensor=kwds.get("convert_to_tensor", False),
                convert_to_numpy=kwds.get("convert_to_numpy", False),
                device=kwds.get("device", None),
                normalize_embeddings=kwds.get("normalize_embeddings", False),
            )

            if not kwds.get("convert_to_tensor") and not kwds.get("convert_to_numpy"):
                embeddings = [embedding.tolist() for embedding in embeddings]

            if len(embeddings) == 1:
                embeddings = embeddings[0]

            return embeddings
        except Exception as e:
            error_explainer(e)
            return {"error": 500, "reason": str(e)}
