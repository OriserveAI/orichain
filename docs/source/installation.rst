Installation
============

We recommend installing orichain with `uv <https://github.com/astral-sh/uv>`_

You can install uv with:

.. code-block:: bash
    
    pip install uv

Then run:

.. code-block:: bash
    
    uv pip install orichain

We have added Sentence Transformers and Lingua Language Detector as optional packages, so if you want to use them, please do one of the following:

For sentence-transformers:

.. code-block:: bash

    uv pip install "orichain[sentence-transformers]"

For lingua-language-detector:

.. code-block:: bash

    uv pip install "orichain[lingua-language-detector]"
