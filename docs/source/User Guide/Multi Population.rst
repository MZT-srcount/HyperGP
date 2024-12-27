Multi-Population Run
===========================================

1. Run multi-popultion with multiple optimizer, using the ``Optimizer.detach`` function:
    + Firstly, we initialize like in **Quick Start**:

    .. code-block:: python
        :linenos:
        
        import random, HyperGP
        from HyperGP import Population, PrimitiveSet, executor, Tensor, GpOptimizer
        from HyperGP.library.operators import RandTrCrv, RandTrMut
        from HyperGP.states import ProgBuildStates, ParaStates
        
        # Generate training set
        input_array = Tensor.uniform(0, 10, size=(2, 10000))
        target = HyperGP.exp((input_array[0] + 1) ** 2) / (input_array[1] + input_array[0])

        # Generate primitive set
        pset = PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])
        
        # Init workflow
        optimizer = GpOptimizer()

        # Add components
        # Add components
        optimizer.iter_component(
            ParaStates(func=RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                        mask=[list(random.sample(range(100), 50)), list(random.sample(range(100), 50))]),
            ParaStates(func=RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), True], to=["p_list"],
                        mask=[random.sample(range(100), 100), 1, 1]),
            ParaStates(func=ExecGPU(), source=["p_list", "input", "pset"], to=["output", None],
                        mask=[1, 1, 1]),
            ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                        mask=[1, 1]))
    
    
    + Here, we initialize multiple population:

    .. code-block:: python
        :linenos:

        pop_num = 5

        optimizers = [optimizer]
        # Init population
        pop = [
            Population(pop_size=100, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)
            for i in range(pop_num)
        ]

        optimizers.append([optimizer.detach() for i in range(pop_num - 1)])

        # Register relevant states
        for i in range(pop_num):
            optimizers[i].status_init(
                p_list=pop[i].states['progs'].indivs,
                input=input_array,pset=pset,output=None,
                fit_list = pop[i].states['progs'].fitness)
                
    + Run multiple optimizers:
    
    .. code-block:: python
        :linenos:

        for i in range(pop_num):
            optimizers[i].run(20)
            fit_best = optimizers[i].fit_list[HyperGP.argsort(optimizers[i].fit_list)[0]]
            print('fit_best: ', fit_best)

       

2. Run multi-population within a single optimizer:
    We can also run the multiple population within a single optimizer, using the mask set:
    
    + add component:
    
    .. code-block:: python
        :linenos:

        def set_mask(pop_num, per_size):
            cdds_1, cdds_2 = [], []
            for i in pop_num:
                cdds_1 += random.sample(range(per_size * i, per_size * (i + 1)), per_size / 2)
                cdds_2 += random.sample(range(per_size * i, per_size * (i + 1)), per_size / 2)
            return [cdds_1, cdds_2]

        # Add components
        optimizer.iter_component(
            ParaStates(func=RandTrCrv(), source=["multip_list", "multip_list"], to=["multip_list", "multip_list"],
                        mask=set_mask(5, 100)),
            ParaStates(func=RandTrMut(), source=["multip_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), True], to=["multip_list"],
                        mask=[random.sample(range(100), 100), 1, 1]),
            ParaStates(func=ExecGPU(), source=["multip_list", "input", "pset"], to=["output", None],
                        mask=[1, 1, 1]),
            ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                        mask=[1, 1]))
        
    
             
    + Intialize the optimizer:

    .. code-block:: python
        :linenos:

        pop_num = 5

        # Init population
        pop = Population(pop_size=500, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)

        # Register relevant states
        optimizer.status_init(
            multip_list=pop.states['progs'].indivs,
            input=input_array,pset=pset,output=None,
            fit_list = pop.states['progs'].fitness)
                
    + Run the optimizer:
    
    .. code-block:: python
        :linenos:

        optimizers.run(20)
        fit_best = [optimizers.fit_list[HyperGP.argsort(optimizers.fit_list[100 * i:100 *(i+1)])[0]] for i in range(pop_num)]
        print('fit_best: ', fit_best)
