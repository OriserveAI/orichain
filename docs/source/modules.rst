Modules
=======

Orichain is built around four core modules, each designed to provide flexibility and simplicity for building Retrieval-Augmented Generation (RAG) workflows. All modules offer both synchronous and asynchronous implementations to suit your application's needs.

**Core Modules Overview:**

- **Embedding Model**  
  Generate vector embeddings from your text data to enable similarity search and retrieval tasks.

- **LLM (Large Language Model)**  
  Interface with large language models for text generation, question answering, tool calling, and more.

- **Knowledge Base**  
  A structured interface for retrieving relevant documents efficiently.

- **Language Detector**  
  Detect the language of user input with configurable options to suit your domain.

----

**API Reference**

Explore the detailed API reference for each module below:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   orichain.embeddings
   orichain.llm
   orichain.knowledge_base
   orichain.lang_detect
