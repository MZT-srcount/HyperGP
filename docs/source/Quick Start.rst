Quick Start
===========================================

Here, We introduce the basic usage of HyperGP

1. **import modoule**: Three types module should be import to run:  
  
   + *basic components*:  
      - ``Population`` to initialize population
      - ``PrimitiveSet`` to set the primitives and terminals
      - ``executor`` to execute the expression
      - ``Tensor`` to store and compute datas
      - ``GpOptimizer`` a workflow manager, to iter overall process 

   + *operators*:
      - such as: ``RandTrCrv``, ``RandTrMut``

   + *states*:
      - such as ``ProgBuildStates``, ``ParaStates``

.. code-block:: python
   :linenos:

   import random, HyperGP, numpy as np
   from HyperGP.states import ProgBuildStates, ParaStates

2. **generate the training data**: We can use ``Tensor`` module to generate the array, or use to encapsulate the ``numpy.ndarray`` or the ``list``

.. code-block:: python
   :linenos:

   # Generate training set
   input_array = HyperGP.tensor.uniform(0, 100, size=(2, input_size))
   target = (input_array[0] + input_array[0] * input_array[1] * input_array[1]) * (input_array[0]) / (input_array[1] + input_array[0])

3. **build the primitive set**: To run the program, we will need  the ``PrimitiveSet`` module to define the used primitives and terminals

.. code-block:: python
   :linenos:

   # Generate primitive set
   pset = HyperGP.PrimitiveSet(input_arity=2,  primitive_set=[('add', HyperGP.tensor.add, 2),('sub', HyperGP.tensor.sub, 2),('mul', HyperGP.tensor.mul, 2),('div', HyperGP.tensor.div, 2),('sin', HyperGP.tensor.sin, 1),('cos', HyperGP.tensor.cos, 1)])

4. **initialize population**: with the ``PrimitiveSet``, we can use ``Population`` to initialize the population
    
.. code-block:: python
   :linenos:

    # Init population
   pop = HyperGP.Population()
   pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100000))
   output, _ = HyperGP.executor(pop.states['progs'].indivs, input=input_array, pset=pset)
   pop.states['progs'].fitness = HyperGP.ops.rmse(output, target)


5. **initialize** ``GpOptimizer`` **workflow module**: To run a workflow, we should first initialize it and set the states we use to the GpOptimizer.

.. code-block:: python
   :linenos:

    # Init workflow
   optimizer = HyperGP.GpOptimizer()
   optimizer.status_init(
      p_list=pop.states['progs'].indivs, target=target,
      input=input_array,pset=pset,output=None,
      fit_list = pop.states['progs'].fitness,
      pp_list=[ind.copy() for ind in pop.states['progs'].indivs],  pfit_list=pop.states['progs'].fitness.copy(),
    )

6. **add the component user want to iteratively run**

.. code-block:: python
   :linenos:

   optimizer.iter_component(
        ParaStates(func=HyperGP.ops.RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                    mask=[lambda x=int(pop_size / 2):random.sample(range(pop_size), x), lambda x=int(pop_size / 2):random.sample(range(pop_size), x)]),
        ParaStates(func=HyperGP.ops.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=pop_size), True], to=["p_list"],
                    mask=[lambda x=pop_size:random.sample(range(pop_size), x), 1, 1]),
        ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=HyperGP.ops.rmse, source=["output", "target"], to=["fit_list"]),
        ParaStates(func=HyperGP.ops.tournament, source=["p_list", "pp_list", "fit_list", "pfit_list"], to=["p_list", "pp_list", "fit_list", "pfit_list"],
                    mask=[1, 1, 1, 1])
    )

8. **run the optimizer**

.. code-block:: python
   :linenos:

   optimizer.monitor(HyperGP.monitors.statistics_record, "fit_list")
   optimizer.run(50, stop_criteria=lambda: HyperGP.tensor.min(optimizer.workflowstates.fit_list) < 1e-9, tqdm_diable=False)
   