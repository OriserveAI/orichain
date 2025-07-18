.. Orichain documentation master file, created by
   sphinx-quickstart on Mon Jul 14 19:36:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Orichain's documentation!
====================================

**Orichain** (/ori'chain/) is a lightweight, Python-based library designed specifically for Retrieval-Augmented Generation (RAG) use cases. Built for seamless integration with your endpoints, Orichain simplifies the process of writing, maintaining, and reviewing RAG workflows.

While inspired by `LangChain <https://www.langchain.com>`_, Orichain focuses on performance and efficiency. Its fully *asynchronous* and *threaded* architecture ensures high concurrency and responsiveness out-of-the-box - so you can focus on building, not optimizing.

.. note::

   - The Library was rewritten in v2.0.0 released in March of 2025. Starting with this version, each module now provides separate synchronous and asynchronous classes, giving you greater flexibility to choose the approach that best fits your application.
   - Starting from version 2.0.1 (May 2025), a `provider` argument is now required when initializing the LLM and EmbeddingModel classes.

----

**Get started here:**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   modules
   code_examples

----

💡 **Additional Resources**

- 📂 **Source Code:** Visit the `Orichain GitHub Repository <https://github.com/OriserveAI/orichain>`_ for reporting issues, exploring the codebase, or contributing to the project.
