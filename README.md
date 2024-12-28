# HyperGP: A high performance heterogeneous parallel GP framework

[![License: BSD 3-Clause License](https://img.shields.io/badge/License-BSD%203--Clause-red)](https://github.com/MZT-srcount/HyperGP/blob/main/LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-support-blue)](https://pypi.org/project/HyperGP/)
[![readthedocs](https://img.shields.io/badge/docs-passing-green)](https://hypergp.readthedocs.io/en/latest/)

PyGP is an open-source high performance framework, providing convenient distributed heterogeneous accleration for the custom prototyping of Genetic Programming (GP) and its variants. To ensure both flexibility and high performance, PyGP encompasses a variety of technologies for GP characteristics to provide convenient prototyping and efficient acceleration of various custom algorithms. To enable quick prototyping within PyGP for research on different types of genetic programming and different application fields, adaptability is also emphasized in building the PyGP framework, to support a wide range of potential applications, such as symbolic regression and image classification. 


## Main Features

A rich acceleration mode are supported.

| **Features**                | **HyperGP** |
| --------------------------- | ----------------------|
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Acceleration for Custom algorithms           | :heavy_check_mark: |
| Acceleration for Custom monitors             | :heavy_check_mark: |
| Acceleration for Custom representation | :heavy_check_mark: |
| Multi-node parallel         | :heavy_check_mark: |
| GPU-Acceleration            | :heavy_check_mark: |
| Hybrid Acceleration with other library   | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |

# Documentation
Documentation is available online: https://hypergp.readthedocs.io/en/latest/

# Installation
HyperGP is available on PyPI and can be installed with:

```
pip install HyperGP
```

Supported Operation Systems: ``Linux``

# Quick Start for Symbolic Regression

1. **import modoule**: Three types module should be import to run:  
  
   - *basic components*:  
      - ``Population`` to initialize population
      - ``PrimitiveSet`` to set the primitives and terminals
      - ``executor`` to execute the expression
      - ``GpOptimizer`` a workflow manager, to iter overall process 

   - *operators*:
      - such as: ``RandTrCrv``, ``RandTrMut``

   - *states*:
      - such as ``ProgBuildStates``, ``ParaStates``

```
    import random, HyperGP
    from HyperGP.states import ProgBuildStates, ParaStates
```

2. **generate the training data**: We can use ``Tensor`` module to generate the array, or use to encapsulate the ``numpy.ndarray`` or the ``list``
```
    # Generate training set
    input_array = HyperGP.uniform(0, 10, size=(2, 10000))
    target = HyperGP.exp((input_array[0] + 1) ** 2) / (input_array[1] + input_array[0])
```
3. **Initialize the basic elements**: To run the program, a ``PrimitiveSet`` module is needed to define the used primitives and terminals, ``Population`` module is used to initialize the population, ``GPOptimizer`` is a workflow used to manage the evolution process.

```
    # Generate primitive set
    pset = HyperGP.PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])
    # Init population
    pop = HyperGP.Population(pop_size=100, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)
    # Init workflow
    optimizer = HyperGP.GpOptimizer()
    # Register relevant states
    optimizer.status_init(
        p_list=pop.states['progs'].indivs,
        input=input_array,pset=pset,output=None,
        fit_list = pop.states['progs'].fitness)
```


4. **build the self-define evaluation function**: Here we use rmse as an example.
```
    def evaluation(output, target):
        r1 = HyperGP.tensor.sub(output, target, dim_0=1)
        return (r1 * r1).sum(dim=1).sqrt()
```

5. **add the component user want to iteratively run**
```
    # Add components
    optimizer.iter_component(
        ParaStates(func=HyperGP.ops.RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                    mask=[list(random.sample(range(100), 50)), list(random.sample(range(100), 50))]),
        ParaStates(func=HyperGP.ops.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), True], to=["p_list"],
                    mask=[random.sample(range(100), 100), 1, 1]),
        ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                    mask=[1, 1]))
```
6. **run the optimizer**
```
    # Iteratively run
    optimizer.run(100)
```

# More User-cases and Applications


| **Example**                | **Link** |
| --------------------------- | ----------------------|
| Example on Symbolic Regression               | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Example on Image Classification        | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Multi-Population Run           | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Multi-task Run             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Custom Representation             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Custom operators             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |
| Hybrid with other libraries             | [![readthedocs](https://img.shields.io/badge/docs-passing-green)]() |