Installation Guide
=============================================

Prerequisites
-----------------------

- Python Version: 3.12+

- Supported Operation Systems: ``Linux``

Binaries
-------------------------

HyperGP is available on PyPI and can be simply installed with:

.. code-block:: python
   :linenos:

    pip install HyperGP

From Source
---------------------

If you are installing from source, you will need:

- Python 3.12 or later
- A compiler that fully supports C++11, such as gcc (gcc 8.5.0 or newer is required, on Linux)
- `NVIDIA CUDA support <https://developer.nvidia.com/cuda-downloads>`_: a driver version higher than 12.2, with a runtime version >= 10.1.105

An example of environment setup in Linux is shown below:

.. code-block:: python
   :linenos:
   
   $ conda env create -n HyperGP -f environment.yml
   $ conda activate HyperGP
   $ cd HyperGP
   $ make all