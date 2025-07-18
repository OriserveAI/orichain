Getting Started
===============

This guide will help you quickly get started with Orichain by demonstrating common use cases for each of its core modules.

----

Creating Embeddings with Orichain
---------------------------------

You can generate embeddings from user queries using the `EmbeddingModel` class.  
Both synchronous and asynchronous variants are available.

.. code-block:: python

    import os
    from dotenv import load_dotenv
    from orichain.embeddings import EmbeddingModel  # Use AsyncEmbeddingModel for async usage

    load_dotenv()

    embedding_model = EmbeddingModel(
        model_name="text-embedding-ada-002",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    query_embedding = embedding_model(user_message="Hey there!")

----

Generating LLM Responses with Orichain
--------------------------------------

Orichain provides easy integration with large language models for text generation tasks.  
Here’s an example using the asynchronous `AsyncLLM` class.

.. code-block:: python

    from orichain.llm import AsyncLLM  # Use LLM for synchronous calls

    llm = AsyncLLM(
        model_name="gpt-4.1-mini",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    response = await llm(
        user_message="Why is the sky blue?"
    )

----

Retrieving Documents with Orichain Knowledge Base
-------------------------------------------------

The `KnowledgeBase` class allows you to retrieve relevant documents from a vector database like Pinecone.

.. code-block:: python

    from orichain.embeddings import EmbeddingModel
    from orichain.knowledge_base import KnowledgeBase

    embedding_model = EmbeddingModel(
        model_name="text-embedding-ada-002",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    knowledge_base = KnowledgeBase(
        vector_db_type="pinecone",
        api_key=os.getenv("PINECONE_KEY"),
        index_name="your index name",
        namespace="your namespace"
    )

    # Create query embeddings for retrieval
    query_embedding = embedding_model(user_message="How to contact your customer support?")

    # Retrieve relevant data chunks
    retrieved_chunks = knowledge_base(
        user_message_vector=query_embedding,
        num_of_chunks=5,
    )

    # Alternatively, fetch documents by IDs
    retrieved_chunks = knowledge_base.fetch(
        ids=["ID_1", "ID_2", "ID_3"]
    )

----

Detecting Language with Orichain
--------------------------------

You can detect the language of user queries using the `LanguageDetection` class.

.. code-block:: python

    from orichain.lang_detect import LanguageDetection

    lang_detect = LanguageDetection(
        languages=["ENGLISH", "ARABIC"],
        min_words=2
    )

    user_language = lang_detect(user_message="هل يمكنك مساعدتي في استفساري؟")

----

Next Steps
----------

Check out the :doc:`code_examples` for more practical demonstrations on how to integrate Orichain into your applications.
