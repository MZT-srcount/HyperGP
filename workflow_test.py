import random, HyperGP
from HyperGP import states
from HyperGP.represent import *
from HyperGP import Population
from HyperGP.states import ProgBuildStates, ParaStates
import numpy as np

# Generate training set
input_array = HyperGP.Tensor(np.random.uniform(0, 10, size=(2, 10000)))
target = HyperGP.exp((input_array[0] + 1) * (input_array[0] + 1)) / (input_array[1] + input_array[0])
print("---------------------")

# Generate primitive set
pset = HyperGP.PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])
# Init population
print("--------0-------------")
pop = HyperGP.Population()
pop.initPop(pop_size=100, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), parallel=False)
# Init workflow
print("--------1-------------")
optimizer = HyperGP.GpOptimizer()
# Register relevant states
print(len(pop.states['progs'].indivs))
optimizer.status_init(
    p_list=pop.states['progs'].indivs, target=target,
    input=input_array,pset=pset,output=None,
    fit_list = pop.states['progs'].fitness)

print("---------2------------")
def evaluation(output, target):
    r1 = HyperGP.tensor.sub(output, target, dim_0=1)
    return (r1 * r1).sum(dim=1).sqrt()

# Add components
print([list(random.sample(range(100), 50)), list(random.sample(range(100), 50))])

def set_prmask(size):
    cdd = random.sample(range(size), size)
    return [[cdd[i] for i in range(0, size, 2)], [cdd[i] for i in range(1, size, 2)]]
# print(set_prmask(100))
# assert 0==1
optimizer.iter_component(
    ParaStates(func=HyperGP.ops.RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                mask=set_prmask(100)),
    ParaStates(func=HyperGP.ops.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=10000), True], to=["p_list"],
                mask=[random.sample(range(100), 100), 1, 1]),
    ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None],
                mask=[1, 1, 1]),
    ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                mask=[1, 1]))

print("---------3------------")
optimizer.run(50)