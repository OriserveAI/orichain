Code Examples
===============

Tool Calling with orichain.llm
---------------------------------

Orichain provides a consistent tool-calling interface in both `LLM` and `AsyncLLM` classes.  
Tool calling works seamlessly with both streaming and non-streaming responses.  

Thanks to the unified design, you only need to change the `model_name`, `provider` and authentication parameters - the rest of your code remains unchanged.

.. code-block:: python

    import os
    from dotenv import load_dotenv
    from orichain.llm import AsyncLLM  # Use LLM for synchronous usage

    load_dotenv()

    tools = [
        {
            "name": "get_weather",
            "description": "Retrieve the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to fetch the weather for.",
                    },
                },
                "required": ["city"],
            }
        }
    ]

    llm = AsyncLLM(
        model_name="gpt-5-mini", 
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY"),
    )

    response = await llm(
        user_message="What’s the weather in Berlin?",
        system_prompt="You are a helpful assistant that provides weather updates.",
        tools=tools,
        tool_choice="auto",  # Options: "auto", "required", "none", or a specific tool name
    )

    print(response.get("tools"))

Output:

.. code-block:: python

    [
        {
            "id": "call_wCYarGvab6vjeTwyHPPuVzod",
            "function": {
                "arguments": {"city": "Berlin"},
                "name": "get_weather"
            }
        }
    ]

----

Using Orichain in a Production-Ready FastAPI Application
--------------------------------------------------------

The following example demonstrates how to integrate Orichain with a FastAPI application for a production-grade setup.  
It shows how to use embeddings, a knowledge base, and an LLM together to build a retrieval-augmented chatbot with both streaming and non-streaming responses.

.. code-block:: python
    
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    from orichain.embeddings import AsyncEmbeddingModel
    from orichain.knowledge_base import AsyncKnowledgeBase
    from orichain.llm import AsyncLLM

    import os
    import art
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

    print(art.text2art("Server has started!", font="small"))
