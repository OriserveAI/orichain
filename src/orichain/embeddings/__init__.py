from typing import Any, List, Dict, Union
from orichain.embeddings import (
    openai_embeddings,
    awsbedrock_embeddings,
    stransformers_embeddings,
    azureopenai_embeddings,
)
import warnings
from orichain import hf_repo_exists


class EmbeddingModels(object):
    """Base class for embedding generation
    Default embedding model that will be used is `text-embedding-ada-002`"""

    default_model = "text-embedding-ada-002"

    model_class = {
        "text-embedding-ada-002": "OpenAI",
        "text-embedding-3-large": "OpenAI",
        "text-embedding-3-small": "OpenAI",
        "amazon.titan-embed-text-v1": "AWSBedrock",
        "amazon.titan-embed-text-v2:0": "AWSBedrock",
        "cohere.embed-english-v3": "AWSBedrock",
        "cohere.embed-multilingual-v3": "AWSBedrock",
    }

    def __init__(self, **kwds: Any) -> None:
        if not kwds.get("model_name"):
            warnings.warn(
                f"No 'model_name' specified, hence defaulting to {self.default_model}",
                UserWarning,
            )

        self.model_name = kwds.get("model_name", self.default_model)

        if kwds.get("use_azure_openai"):
            self.model_type = "AzureOpenAI"
        elif hf_repo_exists(
            repo_id=self.model_name,
            repo_type=kwds.get("repo_type"),
            token=kwds.get("token"),
        ):
            self.model_type = "SentenceTransformer"
        else:
            self.model_type = self.model_class.get(self.model_name)

        if not self.model_type:
            raise ValueError(
                f"Unsupported model: {self.model_name}\n"
                f"Supported models are:\n"
                f"- "
                + "\n- ".join(list(self.model_class.keys()))
                + "\n- All sentence-transformers models"
            )

        model_handler = {
            "OpenAI": openai_embeddings.Embed,
            "AWSBedrock": awsbedrock_embeddings.Embed,
            "SentenceTransformer": stransformers_embeddings.Embed,
            "AzureOpenAI": azureopenai_embeddings.Embed,
        }
        self.model = model_handler.get(self.model_type)(**kwds)

    async def __call__(
        self, user_message: Union[str, List[str]], **kwds: Any
    ) -> Union[List[float], List[List[float]], Dict]:
        if kwds.get("model_name"):
            if self.model_class.get(kwds.get("model_name")) == self.model_type:
                model_name = kwds.get("model_name")
            elif self.model_type == "SentenceTransformer":
                warnings.warn(
                    f"For using different sentence-transformers model: {kwds.get('model_name')}\n"
                    f"again reinitialize the EmbeddingModels class as currently {self.model_name} is already loaded"
                    f"Hence defaulting the model to {self.model_name}",
                    UserWarning,
                )

                model_name = self.model_name
            else:
                warnings.warn(
                    f"Unsupported model: {kwds.get('model_name')}\nSupported models are:"
                    f"\n- "
                    + "\n- ".join(list(self.model_class.keys()))
                    + "\n- All sentence-transformers models\n"
                    f"Hence defaulting to {self.model_name}",
                    UserWarning,
                )
                model_name = self.model_name
        else:
            model_name = self.model_name

        user_message_vector = await self.model(
            text=user_message, model_name=model_name, **kwds
        )

        return user_message_vector
