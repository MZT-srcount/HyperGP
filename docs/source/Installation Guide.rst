Installation Guide
=============================================

Prerequisites
-----------------------

- python_requires=">=3.9, <=3.13"

- Supported Operation Systems: ``Linux``

- [NVIDIA CUDA support](https://developer.nvidia.com/cuda-downloads), with a runtime version >= 10.1.105


Binaries
-------------------------

HyperGP is available on PyPI and can be simply installed with:

.. code-block:: bash
   :linenos:

    pip install HyperGP

From Source
---------------------

If you are installing from source, you will need:

- Python 3.12 or later
- A compiler that fully supports C++11, such as gcc (gcc 8.5.0 or newer is required, on Linux)
- `NVIDIA CUDA support <https://developer.nvidia.com/cuda-downloads>`_: a driver version higher than 12.2, with a runtime version >= 10.1.105

An example of environment setup in Linux is shown below:

- You should create a conda environment with some dependencies first:

.. code-block:: bash
   :linenos:
   
   $ conda env create -n HyperGP -f environment.yml
   $ conda activate HyperGP

- If you want to build a wheel in local, just run the following command:

.. code-block:: bash
   :linenos:
   
   $ python ./setup.py sdist bdist_wheel

- Then, you can use the ``HyperGP`` through the ``pip install`` (replace the xxx to the actual string of the whl in ``/dist``):

.. code-block:: bash
   :linenos:

   $ pip install HyperGP-xxx.whl

- Or you can directly directly run the ``HyperGP`` through the source code, with the following command:

.. code-block:: bash
   :linenos:

   $ cd HyperGP
   $ make compile
   $ cd ..
   
   
