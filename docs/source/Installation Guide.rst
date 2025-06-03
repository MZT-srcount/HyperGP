Installation Guide
=============================================

Prerequisites
-----------------------

- python_requires=">=3.9, <=3.13"

- Supported Operation Systems: ``Linux``

- [NVIDIA CUDA support](https://developer.nvidia.com/cuda-downloads), with a runtime version >= 11.4.2 to support asynchronize stream computation


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
- `NVIDIA CUDA support <https://developer.nvidia.com/cuda-downloads>`_:, with a runtime version >= 11.4.2

An example of environment setup in Linux is shown below:

- You should create a conda environment with some dependencies first:

.. code-block:: bash
   :linenos:
   
   conda env create -n HyperGP -f environment.yml
   conda activate HyperGP

- If you want to build a wheel in local, just run the following command:

.. code-block:: bash
   :linenos:
   
   python ./setup.py sdist bdist_wheel

- Then, you can use the ``HyperGP`` through the ``pip install`` (replace the ``{str}`` to the actual string of the whl in ``/dist``):

.. code-block:: bash
   :linenos:

   pip install HyperGP-{str}.whl

- Or you can directly directly run the ``HyperGP`` through the source code, with the following command:

.. code-block:: bash
   :linenos:

   cd HyperGP
   make compile
   cd ..


- create a conda environment and install the dependencies by running the following command:

.. code-block:: bash
   :linenos:

   conda env create -n HyperGP -f environment.yml
   conda activate HyperGP

- Activate the c++ compiler:

.. code-block:: bash
   :linenos:

   conda env config vars set CC=x86_64-conda-linux-gnu-gcc
   conda env config vars set CXX=x86_64-conda-linux-gnu-g++
   conda deactivate

- Build HyperGP:

.. code-block:: bash
   :linenos:

   conda activate HyperGP
   cd HyperGP
   rm -rf ./build/*
   make compile
   cd ..
   
   
