Code Examples
===============

Here is a basic example of how to use this code in a FastAPI application:

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

    embedding_model = AsyncEmbeddingModel(
        model_name="text-embedding-ada-002",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    knowledge_base_manager = AsyncKnowledgeBase(
        vector_db_type="pinecone",
        api_key=os.getenv("PINECONE_KEY"),
        index_name="<depends on your creds>", 
        namespace="<choose your desired namespace",
    )

    llm = AsyncLLM(
        model_name="gpt-4.1-mini", 
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    app = FastAPI(redoc_url=None, docs_url=None)

    @app.post("/generative_response")
    async def generate(request: Request) -> Response:
        # Fetching data from the request recevied
        request_json = await request.json()

        # Fetching valid keys
        user_message = request_json.get("user_message")
        prev_pairs = request_json.get("prev_pairs")
        metadata = request_json.get("metadata")

        # Embedding creation for retrieval
        user_message_vector = await embedding_model(user_message=user_message)

        # Checking for error while embedding generation
        if isinstance(user_message_vector, Dict):
            return JSONResponse(user_message_vector)

        # Fetching relevant data chunks from knowledgebase
        retrived_chunks = await knowledge_base_manager(
            user_message_vector=user_message_vector,
            num_of_chunks=5,
        )

        # Checking for error while fetching relevant data chunks
        if isinstance(retrived_chunks, Dict) and "error" in retrived_chunks:
            return JSONResponse(user_message_vector)

        matched_sentence = convert_to_text_list(retrived_chunks) # Create a funtion that converts your data into a list of relevant information

        system_prompt = f"""As a helpful, engaging and friendly chatbot, you will answer the user's query based on the following information provided: 
        <data>
        {"\n\n".join(matched_sentence)}
        </data>"""

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

    print(art.text2art("Server has started!", font="small"))
