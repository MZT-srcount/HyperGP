# Quick Start

```python
import time
import random, HyperGP, numpy as np
from HyperGP.states import ProgBuildStates, ParaStates

np.random.seed(0)
random.seed(0)

pop_size = 1000
input_size = 10000

input_array = HyperGP.tensor.uniform(0, 10, size=(2, input_size))

target = (input_array[0] * 3 + input_array[0] * input_array[1] * 2) * (input_array[0])

pset = HyperGP.PrimitiveSet(input_arity=2,  primitive_set=[('add', HyperGP.tensor.add, 2),('sub', HyperGP.tensor.sub, 2),('mul', HyperGP.tensor.mul, 2),('div', HyperGP.tensor.div, 2),('sin', HyperGP.tensor.sin, 1),('cos', HyperGP.tensor.cos, 1)])

pop = HyperGP.Population()
pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100000))
output, _ = HyperGP.executor(pop.states['progs'].indivs, input=input_array, pset=pset)
pop.states['progs'].fitness = HyperGP.ops.rmse(output, target)

optimizer = HyperGP.GpOptimizer()
optimizer.status_init(
    childs_list=pop.states['progs'].indivs, target=target,
    input=input_array,pset=pset,output=None,
    cfit_list = pop.states['progs'].fitness,
    parent_list=[ind.copy() for ind in pop.states['progs'].indivs],  pfit_list=pop.states['progs'].fitness.copy(),
    )

optimizer.iter_component(
        ParaStates(func=HyperGP.ops.RandTrCrv(), source=["parent_list", "parent_list", 0.8], to=["parent_list", "parent_list"],
                    mask=[lambda x=int(pop_size / 2):random.sample(range(pop_size), x), lambda x=int(pop_size / 2):random.sample(range(pop_size), x), 1]),
        ParaStates(func=HyperGP.ops.RandTrMut(), source=["parent_list", ProgBuildStates(pset=pset, depth_rg=[1, 3], len_limit=pop_size), 0.2, True], to=["parent_list"],
                    mask=[lambda x=pop_size:random.sample(range(pop_size), x), 1, 1, 1]),
        ParaStates(func=HyperGP.executor, source=["parent_list", "input", "pset"], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=HyperGP.ops.rmse, source=["output", "target"], to=["cfit_list"]),
        ParaStates(func=HyperGP.ops.tournament, source=["parent_list", "parent_list", "cfit_list", "pfit_list"], to=["parent_list", "parent_list", "cfit_list", "pfit_list"],
                    mask=[1, 1, 1, 1]),
    )

optimizer.monitor(HyperGP.monitors.statistics_record, "cfit_list")
optimizer.run(100, stop_criteria=lambda: HyperGP.tensor.min(optimizer.workflowstates.cfit_list) == 0.0, tqdm_diable=False)

```