Getting Started
===============

Here's how you can get started easily with Orichain:

Creating Embeddings using Orichain
-----------------------------------

Here is a basic example of how to use EmbeddingModel class from Orichain:

.. code-block:: python
    
    import os
    from orichain.embeddings import EmbeddingModel # Use AsyncEmbeddingModel for asynchronous embeddings creation
    from dotenv import load_dotenv

    load_dotenv()

    embedding_model = EmbeddingModel(
        model_name="text-embedding-ada-002",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    query_embedding = embedding_model(user_message="Hey there!")
    

Using Orichain to generate responses from an LLM
-------------------------------------------------

Let's use AsyncLLM class from Orichain to generate a response asynchronously:

.. code-block:: python
    
    from orichain.llm import AsyncLLM # Use LLM for synchronous response generation

    llm = AsyncLLM(
        model_name="gpt-4.1-mini", 
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    response = await llm(
        user_message="Why is the sky Blue?",
    )


Using Knowledge Base from Orichain
------------------------------------------------

Let's use KnowledgeBase class from Orichain to retrieve relevant documents from pinecone:

.. code-block:: python
    
    from orichain.embeddings import EmbeddingModel
    from orichain.knowledge_base import KnowledgeBase

    embedding_model = EmbeddingModel(
        model_name="text-embedding-ada-002",
        provider="OpenAI",
        api_key=os.getenv("OPENAI_KEY")
    )

    knowledge_base_manager = KnowledgeBase(
        vector_db_type="pinecone",
        api_key=os.getenv("PINECONE_KEY"),
        index_name="<your index name here>", 
        namespace="<your desired namespace",
    )

    # Embedding creation for retrieval
    query_embedding = embedding_model(user_message="How to contact your customer support?")

    # Querying relevant data chunks from knowledgebase
    retrieved_chunks = knowledge_base_manager(
        user_message_vector=query_embedding,
        num_of_chunks=5,
    )

    # We can also fetch based on document ids
    retrieved_chunks = knowledge_base_manager.fetch(
        ids=["ID_1", "ID_2", "ID_3"]
    )

Using Language Detector from Orichain
------------------------------------------------

Let's use LanguageDetection class from Orichain to detect language of a user query:

.. code-block:: python
    
    from orichain.lang_detect import LanguageDetection

    lang_detect = LanguageDetection(languages=["ENGLISH", "ARABIC"], min_words=2)

    user_language = lang_detect(user_message="هل يمكنك مساعدتي في استفساري؟")

