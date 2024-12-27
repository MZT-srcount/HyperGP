# Quick Start

```python
import random, HyperGP
from HyperGP import Population, PrimitiveSet, ExecGPU, Tensor, GpOptimizer
from HyperGP.library.operators import RandTrCrv, RandTrMut
from HyperGP.states import ProgBuildStates, ParaStates

# Set Mask
def set_prmask(size):
    cdd = random.sample(range(size), size)
    return [[cdd[i] for i in range(0, size, 2)], [cdd[i] for i in range(1, size, 2)]]
def evaluation(output, target):
    r1 = HyperGP.sub(output, target, dim_0=0)
    return (r1 * r1).sum(dim=0).sqrt()

# Generate training set
input_array = Tensor.uniform(0, 10, size=(2, 10000))
target = HyperGP.exp((input_array[0] + 1) ** 2) / (input_array[1] + input_array[0])

# Generate primitive set
pset = PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])

# Init population
pop = Population(pop_size=100, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)

# Init workflow
optimizer = GpOptimizer()

# Register states
optimizer.status_init(
    p_list=pop.states['progs'].indivs,
    input=input_array,pset=pset,output=None,
    fit_list = pop.states['progs'].fitness)

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

# Iteratively run
optimizer.run(10)

```