# HyperGP: A high performance heterogeneous parallel GP framework


[![License: BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-red)](https://github.com/MZT-srcount/HyperGP/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-support-blue)](https://pypi.org/project/HyperGP/)
[![readthedocs](https://img.shields.io/badge/docs-passing-green)](https://hypergp.readthedocs.io/en/latest/)
[![coverage](https://img.shields.io/badge/coverage-passing-green)]()


HyperGP is an open-source high-performance framework, that provides convenient distributed heterogeneous acceleration for the custom prototyping of Genetic Programming (GP) and its variants. To ensure both flexibility and high performance, HyperGP encompasses a variety of technologies for GP characteristics to provide convenient prototyping and efficient acceleration of various custom algorithms. To enable quick prototyping within HyperGP for research on different types of genetic programming and different application fields, adaptability is also emphasized in building the HyperGP framework, to support a wide range of potential applications, like symbolic regression. 

<p align="center">
  <a href="https://hypergp.readthedocs.io/en/latest/Quick%20Start.html">Main Features</a> |
  <a href="https://github.com/MZT-srcount/HyperGP?tab=readme-ov-file#installation">Installation</a> |
  <a href="https://hypergp.readthedocs.io/en/latest/Quick%20Start.html">Documentation</a> |
  <a href="https://github.com/MZT-srcount/HyperGP?tab=readme-ov-file#quick-start-for-symbolic-regression">Examples</a> |
  <a href="https://github.com/MZT-srcount/HyperGP?tab=readme-ov-file#call-for-contributions">Contributing</a> |
</p>

## Main Features

A rich acceleration mode are supported.

| **Features**                | **HyperGP** |
| --------------------------- | ----------------------|
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Acceleration for Custom algorithms           | :heavy_check_mark: |
| Support for Custom monitors             | :heavy_check_mark: |
| Support for Custom representation | :heavy_check_mark: |
| Multi-node parallel         | :heavy_check_mark: |
| GPU-Acceleration            | :heavy_check_mark: |
| Hybrid Acceleration with other library   | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |

# Documentation
Documentation is available online: https://hypergp.readthedocs.io/en/latest/Quick%20Start.html  
- Due to time constraints, the document has not yet been fully completed and is currently being updated continuously.  
# Installation

## Prerequisites

- python_requires=">=3.9, <=3.13"
- Supported Operation Systems: ``Linux``

## Binaries

``HyperGP`` is available on PyPI and can be simply installed with:

```bash
pip install HyperGP
```

Supported Operation Systems: ``Linux``

## From Source

If you are installing from source, you will need:

- A compiler that fully supports C++11, such as gcc (gcc 8.5.0 or newer is required, on Linux)
- [NVIDIA CUDA support](https://developer.nvidia.com/cuda-downloads), with a runtime version >= 11.4.2 to support asynchronize stream computation

An example of environment setup in Linux is shown below:

- create a conda environment and install the dependencies by running the following command:
```bash
$ conda env create -n HyperGP -f environment.yml
$ conda activate HyperGP
```

- Activate the c++ compiler:
```bash
$ conda env config vars set CC=x86_64-conda-linux-gnu-gcc
$ conda env config vars set CXX=x86_64-conda-linux-gnu-g++
$ conda deactivate
```

- Build HyperGP:
```bash
$ conda activate HyperGP
$ cd HyperGP
$ rm -rf ./build/*
$ make compile
$ cd ..
```

# Quick Start for Symbolic Regression

## Run the examples

There are already some simple examples at the [examples](./examples/) directory, you can directly run these examples after installing ``HyperGP`` and activating the environments.

```bash
python ./examples/workflow_test.py
```

## Build a new example

1. **Import modoule**: Three types module should be import to run:  
  
   - *basic components*:  
      - ``Population`` to initialize population
      - ``PrimitiveSet`` to set the primitives and terminals
      - ``executor`` to execute the expression
      - ``GpOptimizer`` a workflow manager, to iter overall process 

   - *operators*:
      - such as: ``RandTrCrv``, ``RandTrMut``

   - *states*:
      - such as ``ProgBuildStates``, ``ParaStates``

```python
    import random, HyperGP, numpy as np
    from HyperGP.states import ProgBuildStates, ParaStates
```

2. **Generate the training data**: We can use ``Tensor`` module to generate the array, or use to encapsulate the ``numpy.ndarray`` or the ``list``

```python
    pop_size = 1000
    input_size = 10000
    # Generate training set
    input_array = HyperGP.tensor.uniform(0, 10, size=(2, input_size))
    target = (input_array[0] * 3 + input_array[0] * input_array[1] * 2) * (input_array[0])

```
3. **Initialize the basic elements**: To run the program, a ``PrimitiveSet`` module is needed to define the used primitives and terminals, ``Population`` module is used to initialize the population, ``GPOptimizer`` is a workflow used to manage the evolution process.

```python
    # Generate primitive set
    pset = HyperGP.PrimitiveSet(input_arity=2,  primitive_set=[('add', HyperGP.tensor.add, 2),('sub', HyperGP.tensor.sub, 2),('mul', HyperGP.tensor.mul, 2),('div', HyperGP.tensor.div, 2),('sin', HyperGP.tensor.sin, 1),('cos', HyperGP.tensor.cos, 1)])
    # Init population
    pop = HyperGP.Population()
    pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100000))
    output, _ = HyperGP.executor(pop.states['progs'].indivs, input=input_array, pset=pset)
    pop.states['progs'].fitness = HyperGP.ops.rmse(output, target)
    # Init workflow
    optimizer = HyperGP.GpOptimizer()
    # Register relevant states
    optimizer.status_init(
        p_list=pop.states['progs'].indivs, target=target,
        input=input_array,pset=pset,output=None,
        fit_list = pop.states['progs'].fitness,
        pp_list=[ind.copy() for ind in pop.states['progs'].indivs],  pfit_list=pop.states['progs'].fitness.copy(),
    )
```


4. **Add the component user want to iteratively run**

```python
    # Add components
    
    optimizer.iter_component(
        ParaStates(func=HyperGP.ops.RandTrCrv(), source=["p_list", "p_list", 0.9], to=["p_list", "p_list"],
                    mask=[lambda x=int(pop_size / 2):random.sample(range(pop_size), x), lambda x=int(pop_size / 2):random.sample(range(pop_size), x), 1]),
        ParaStates(func=HyperGP.ops.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[1, 3], len_limit=pop_size), 0.1, True], to=["p_list"], mask=[lambda x=pop_size:random.sample(range(pop_size), x), 1, 1, 1]),
        ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None], mask=[1, 1, 1]),
        ParaStates(func=HyperGP.ops.rmse, source=["output", "target"], to=["fit_list"]),
        ParaStates(func=HyperGP.ops.tournament, source=["p_list", "pp_list", "fit_list", "pfit_list"], to=["p_list", "pp_list", "fit_list", "pfit_list"], mask=[1, 1, 1, 1]),
    )

```
6. **Run the optimizer**

```python
    # Iteratively run
    optimizer.monitor(HyperGP.monitors.statistics_record, "fit_list")
    optimizer.run(100, stop_criteria=lambda: HyperGP.tensor.min(optimizer.workflowstates.fit_list) < 1e-15, tqdm_diable=False)

```

# More User-cases and Applications


| **Example**                | **Doc Link** | **Runable Examples** |
| --------------------------- | ----------------------| ----------------------|
| Example on Symbolic Regression               | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Symbolic Regression]()|
| Example on Image Classification        | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Image Classification]()|
| Multi-Population Run           | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Multi-Population]()|
| Multi-task Run             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Multi-task]()|
| Custom Representation             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Custom Representation]()|
| Custom operators             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Custom operators]()|
| Hybrid with other libraries             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() | [Hybrid Programming]()|

# Call for Contributions
