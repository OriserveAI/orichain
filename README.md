# 🔗 ORICHAIN

**Orichain** is a Python library designed as a custom wrapper for Retrieval-Augmented Generation (RAG) use cases, built to integrate seamlessly with your endpoints.

It provides support for:

- Embedding creation
    - AWS Bedrock
        - Cohere embeddings
        - Titian embeddings
    - OpenAI Embeddings
    - TogetherAI Embedding Models
    - Google Gemini & Vertex AI Embeddings
    - Azure OpenAI Embeddings
    - Sentence Transformers

- Knowledge base (Vector Databases)
    - Pinecone
    - ChromaDB

- Large Language Models
    - OpenAI
    - Azure OpenAI
    - Anthropic
    - Google Gemini & Vertex AI Models
    - TogetherAI
    - AWS Bedrock
        - Anthropic models (Series 3, 3.5, 3.7, 4)
        - LLAMA models (Series 3, 3.1, 3.2, 3.3, 4)
        - Amazon Titan text models
        - Amazon Nova series models
        - Mistral models
        - Inference Profiles

This library was built to make the applications of all the codes easy to write and review. 
It can be said that it was inspired by LangChain but is optimized for better performance. The entire codebase is asynchronous and threaded, eliminating the need for you to worry about optimization.

## 📚 Documentation
Full documentation is available here:
➡️ https://orichain.readthedocs.io/en/latest/

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## 📦 Installation

> 🚧 Note:
> - The Library was rewritten in v2.0.0 released in March of 2025. There were significant changes made.
> - Starting from version 2.0.1 (May 2025), a `provider` argument is now required when initializing the LLM and EmbeddingModel classes.

Install via pip:
```bash
pip install orichain
```

We have added Sentence Transformers and Lingua Language Detector as optional packages, so if you want to use them, please do one of the following:

#### Installation Options

##### 1️⃣ Install with orichain:
- For `sentence-transformers`:
```bash
pip install "orichain[sentence-transformers]"
```
- For `lingua-language-detector`:
```bash
pip install "orichain[lingua-language-detector]"
```

##### 2️⃣ Install directly:
- For `sentence-transformers`:
```bash
pip install sentence-transformers==3.4.1
```
- For `lingua-language-detector`:
```bash
pip install lingua-language-detector==2.1.0
```

## 🚀 Usage Example

A quick example of how to use Orichain:

```python
from orichain.llm import AsyncLLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = AsyncLLM(
    model_name="gpt-5-mini", 
    provider="OpenAI",
    api_key=os.getenv("OPENAI_KEY")
)

user_message = "I am feeling sad"

system_prompt = """You need to return a JSON object with a key emotion and detect the user emotion like this:
{
    "emotion": return the detected emotion of user
}"""

llm_response = await llm(
    user_message=user_message,
    system_prompt=system_prompt,
    do_json=True # This insures that the response will be a json
)
```

## ✨ Features

Reasons to use Orichain:
- Supports Both Sync and Async: Provides flexibility for different use cases.
- Optimized Performance: Utilizes threading for efficiency, especially when used with FastAPI.
- Hot-Swappable Components: Easily modify RAG components as project requirements evolve, ensuring high flexibility.
- Unified tool-calling interface across all providers, ensuring seamless modularity.

## 💡 Example

Here is a basic example of how to use this code: 

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from orichain.embeddings import AsyncEmbeddingModel
from orichain.knowledge_base import AsyncKnowledgeBase
from orichain.llm import AsyncLLM

import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

# Initialize embedding model
embedding_model = AsyncEmbeddingModel(
    model_name="text-embedding-ada-002",
    provider="OpenAI",
    api_key=os.getenv("OPENAI_KEY")
)

# Initialize vector database manager
knowledge_base_manager = AsyncKnowledgeBase(
    vector_db_type="pinecone",
    api_key=os.getenv("PINECONE_KEY"),
    index_name="<set your index name>", 
    namespace="<set your namespace>",
)

# Initialize LLM
llm = AsyncLLM(
    model_name="gpt-5-mini", 
    provider="OpenAI",
    api_key=os.getenv("OPENAI_KEY")
)

app = FastAPI(redoc_url=None, docs_url=None)

@app.post("/generative_response")
async def generate(request: Request) -> Response:
    # Parse incoming request
    request_json = await request.json()

    user_message = request_json.get("user_message")
    prev_pairs = request_json.get("prev_pairs")
    metadata = request_json.get("metadata")

    # Generate embeddings for the user query
    user_message_vector = await embedding_model(user_message=user_message)

    if isinstance(user_message_vector, Dict):
        return JSONResponse(user_message_vector)

    # Retrieve relevant chunks from the knowledge base
    retrived_chunks = await knowledge_base_manager(
        user_message_vector=user_message_vector,
        num_of_chunks=5,
    )

    if isinstance(retrived_chunks, Dict) and "error" in retrived_chunks:
        return JSONResponse(retrived_chunks)

    # Convert retrieved data into plain text list
    matched_sentence = convert_to_text_list(retrived_chunks)  
    # (Define `convert_to_text_list` to process KB output into a list of strings)

    system_prompt = f"""As a helpful, engaging, and friendly chatbot, answer the user's query based on the following context:
    <data>
    {"\n\n".join(matched_sentence)}
    </data>"""

    # Streaming response
    if metadata.get("stream"):
        return StreamingResponse(
            llm.stream(
                request=request,
                user_message=user_message,
                matched_sentence=matched_sentence,
                system_prompt=system_prompt,
                chat_hist=prev_pairs
            ),
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            media_type="text/event-stream",
        )
    # Non-streaming response
    else:
        llm_response = await llm(
            request=request,
            user_message=user_message,
            matched_sentence=matched_sentence,
            system_prompt=system_prompt,
            chat_hist=prev_pairs
        )

        return JSONResponse(llm_response)
```

## 🛣️ Roadmap

Here's our plan for upcoming features and improvements:

### Short-term goals
- [X] Do testing of the latest version
- [X] Release stable 1.0.0 version
- [X] Create Documentation
- [X] Write class and function definitions
- [X] Add function calling support
- [ ] Add Together AI support

### Long-term goals
- [X] Publish it to pypi
- [X] Refactor the code for better readability

## 🤝 Contributing

We welcome contributions to help us achieve these goals!

### 📝 Steps
1. Stage all changes
```bash
git add .
```

2. Commit the changes
```bash
git commit -m "Release vX.X.X"
```

3. Create a new tag
```bash
git tag vX.X.X
```

4. Push commits to main branch
```bash
git push origin main
```

5. Push the new tag
```bash
git push origin vX.X.X
```

### Deleting Tags:
- Delete a Local Tag
```bash
git tag -d vX.X.X
```

- Delete a Remote Tag
```bash
git push origin --delete vX.X.X
```

## 📄 License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
