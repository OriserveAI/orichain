# ORICHAIN

It is a custom wrapper made for RAG use cases made to be integrated with your endpoints, It caters:

- Embedding creation
    - AWS Bedrock
        - Cohere embeddings
        - Titian embeddings
    - OpenAI Embeddings
    - Azure OpenAI Embeddings
    - Sentence Transformers

- Knowledge base (Vector Databases)
    - Pinecone
    - ChromaDB

- Large Langauge Models
    - OpenAI
    - Azure OpenAI
    - Anthropic
    - AWS Bedrock
        - Anthropic models (Series 3, 3.5)
        - LLAMA models (Series 3, 3.1, 3.2)
        - Amazon Titan text models
        - Mistral models
        - Inference Profiles

This library was built to make the applications of all the codes easy to write and review. 
It can be said it was inspired by langchain but better at optimisation, the whole code is async and threadded, so that you do not need to worry about optimistaion anymore.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Documentation](#documentation)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Installation

Just do this
```
pip install orichain
```

OR

You can also install the wheel file from the main branch of Repo:
https://github.com/OriserveAI/orichain

Copy the dist folder and it should contain these files:
```
- dist
    |-orichain-<version name>-py3-none-any.whl
    |-orichain-<version name>.tar.gz
```

Create a requirements.txt, it should look like this:

```
...
orichain @ file:///<your absoulute path>/dist/orichain-<version name>-py3-none-any.whl
...
```

Then install the library using this command:

```
pip install -r requirements.txt
```

## Usage

A quick example of how to use orichain:

```python
from orichain.llm import LLM
import os
from dotenv import load_dotenv

load_dotenv()

llm = LLM(api_key=os.getenv("OPENAI_KEY"))

user_message = "I am feeling sad"

system_prompt = """You need to return a JSON object with a key emotion and detect the user emotion like this:
{
    "emotion": return the detected emotion of user
}"""

llm_response = await llm(
                request=request, # Request of endpoint when using Fastapi, checks whether the request has been aborted
                user_message=user_message,
                system_prompt=system_prompt,
                do_json=True # This insures that the response wil be a json
            )
```

## Features

Reasons to use orichain:

- Optimized: The whole code is async and parts of it is also threadded, you will be using fastapi so the code will be highly efficient
- Hot Swappable: You can easily change the parts of RAG, whenever the requirements change of the project. Highly flexiable

## Documentation

Coming soon...

## Example

I will give you a basic example of how to use this code

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from orichain.embeddings import EmbeddingModels
from orichain.knowledge_base import KnowledgeBase
from orichain.llm import LLM

import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

embedding_model = EmbeddingModels(api_key=os.getenv("OPENAI_KEY"))

knowledge_base_manager = KnowledgeBase(
    type="pinecone",
    api_key=os.getenv("PINECONE_KEY"),
    index_name="<depends on your creds>", 
    namespace="<choose your desired namespace",
)

llm = LLM(api_key=os.getenv("OPENAI_KEY"))

app = FastAPI(redoc_url=None, docs_url=None)

@app.post("/generative_response")
async def generate(request: Request) -> Response:
    # Fetching data from the request recevied
    request_json = await request.json()

    # Fetching valid keys
    user_message = request_json.get("user_message")
    prev_pairs = request_json.get("prev_pairs")

    # Embedding creation for retrieval
    user_message_vector = await embedding_model(user_message=user_message)

    # Checking for error while embedding generation
    if isinstance(user_message_vector, Dict):
        return JSONResponse(user_message_vector)

    # Fetching relevant data chunks from knowledgebase
    retrived_chunks = await knowledge_base_manager(
        user_message_vector=user_message_vector,
        num_of_chunks=parameters.num_of_chunks,
    )

    # Checking for error while fetching relevant data chunks
    if isinstance(retrived_chunks, Dict) and "error" in retrived_chunks:
        return JSONResponse(user_message_vector)

    matched_sentence = convert_to_text_list(retrived_chunks) # Create a funtion that converts your data into a list of relevant information

    # Streaming
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
    # Non streaming
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

## Roadmap

Here's our plan for upcoming features and improvements:

### Short-term goals
- [X] Do testing of the latest version
- [X] Release stable 1.0.0 version
- [ ] Create Documentation
- [ ] Write class and function definations

### Long-term goals
- [X] Publish it to pypi
- [ ] Refactor the code for better readablity

## Contributing

We welcome contributions to help us achieve these goals!

## License

It is restricted to Oriserve, do not use it any where else but if you do use it anywhere else just do not let me know.