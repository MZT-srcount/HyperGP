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

    import random, HyperGP
    from HyperGP import Population, PrimitiveSet, executor, Tensor, GpOptimizer
    from HyperGP.library.operators import RandTrCrv, RandTrMut
    from HyperGP.states import ProgBuildStates, ParaStates

2. **generate the training data**: We can use ``Tensor`` module to generate the array, or use to encapsulate the ``numpy.ndarray`` or the ``list``

.. code-block:: python
   :linenos:

    # Generate training set
    input_array = Tensor.uniform(0, 10, size=(2, 10000))
    target = HyperGP.exp((input_array[0] + 1) ** 2) / (input_array[1] + input_array[0])

3. **build the primitive set**: To run the program, we will need  the ``PrimitiveSet`` module to define the used primitives and terminals

.. code-block:: python
   :linenos:

    # Generate primitive set
    pset = PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])

4. **initialize population**: with the ``PrimitiveSet``, we can use ``Population`` to initialize the population
    
.. code-block:: python
   :linenos:

    # Init population
    pop = Population(pop_size=100, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)

5. **initialize** ``GpOptimizer`` **workflow module**: To run a workflow, we should first initialize it and set the states we use to the GpOptimizer.

.. code-block:: python
   :linenos:

    # Init workflow
    optimizer = GpOptimizer()

    # Register relevant states
    optimizer.status_init(
        p_list=pop.states['progs'].indivs,
        input=input_array,pset=pset,output=None,
        fit_list = pop.states['progs'].fitness)


6. **build the evaluation function**

.. code-block:: python
   :linenos:

    def evaluation(output, target):
        r1 = HyperGP.sub(output, target, dim_0=1)
        return (r1 * r1).sum(dim=1).sqrt()


7. **set mask**

.. code-block:: python
   :linenos:
   
    # Set Mask
    def set_prmask(size):
        cdd = random.sample(range(size), size)
        return [[cdd[i] for i in range(0, size, 2)], [cdd[i] for i in range(1, size, 2)]]

8. **add the component user want to iteratively run**

.. code-block:: python
   :linenos:

    # Add components
    optimizer.iter_component(
        ParaStates(func=RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                    mask=set_prmask(100)),
        ParaStates(func=RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), True], to=["p_list"],
                    mask=[random.sample(range(100), 100), 1, 1]),
        ParaStates(func=ExecGPU(), source=["p_list", "input", "pset"], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                    mask=[1, 1]))

9. **run the optimizer**

.. code-block:: python
   :linenos:

   optimizer.
    # Iteratively run
    optimizer.run(100)