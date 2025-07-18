Installation
============

Orichain can be installed easily using either `uv <https://github.com/astral-sh/uv>`_ or `pip`. We recommend using **uv** for faster and more reliable dependency resolution.

----

**Install with uv (Recommended)**

First, install **uv**:

.. code-block:: bash

    pip install uv

Then, install Orichain:

.. code-block:: bash

    uv pip install orichain

----

**Install with pip**

Alternatively, you can install Orichain using pip:

.. code-block:: bash

    pip install orichain

----

**Optional Dependencies**

Orichain supports optional integrations with **Sentence Transformers** and **Lingua Language Detector**. If your use case requires these, you can install them as follows:

**For Sentence Transformers support:**

.. code-block:: bash

    uv pip install "orichain[sentence-transformers]"

**For Lingua Language Detector support:**

.. code-block:: bash

    uv pip install "orichain[lingua-language-detector]"

You can also use `pip install` instead of `uv pip install` for these optional dependencies if preferred.

----

**Next Steps:**
After installation, refer to the :doc:`getting_started` guide to learn how to integrate Orichain into your projects.
