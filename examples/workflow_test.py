'''
Descripttion: 
version: 
Author: sueRimn
Date: 2024-12-31 13:28:21
LastEditors: sueRimn
LastEditTime: 2025-01-02 10:13:04
'''
import random, HyperGP, numpy as np
from HyperGP.states import ProgBuildStates, ParaStates


pop_size = 1000
input_size = 10000

input_array = HyperGP.Tensor(np.random.uniform(0, 10, size=(2, input_size)))
target = HyperGP.tensor.exp((input_array[0]) * (input_array[0])) / (input_array[1] + input_array[0])

pset = HyperGP.PrimitiveSet(input_arity=2,  primitive_set=[('add', HyperGP.add, 2),('sub', HyperGP.sub, 2),('mul', HyperGP.mul, 2),('div', HyperGP.div, 2)])
pop = HyperGP.Population()
pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100000))

def evaluation(output, target):
    r1 = HyperGP.tensor.sub(output, target, dim_0=1)
    return (r1 * r1).sum(dim=1).sqrt()

output, _ = HyperGP.executor(pop.states['progs'].indivs, input=input_array, pset=pset)
fitness = evaluation(output, target)

optimizer = HyperGP.GpOptimizer()
optimizer.status_init(
    p_list=pop.states['progs'].indivs, target=target,
    input=input_array,pset=pset,output=None,
    fit_list = pop.states['progs'].fitness,
    pp_list=[ind.copy() for ind in pop.states['progs'].indivs],  pfit_list=pop.states['progs'].fitness.copy(),
    )

def selection(p1, p2, f1, f2):
    p_list, f_list = p1 + p2, HyperGP.tensor.concatenate((f1, f2))
    legal_list = [z for z, prog in enumerate(p_list) if len(prog) < 100]
    sample_list = [list(random.sample(legal_list, 3)) for i in range(len(p1) - 1)]
    tour_list = [int(HyperGP.tensor.argmin(f_list[legal_list]))] + [x[int(HyperGP.tensor.argmin(f_list[x]))] for x in sample_list]
    p_new, f_new = [p_list[sample] for sample in tour_list], f_list[tour_list]
    return p_new, [ind.copy() for ind in p_new], f_new, f_new.copy()


optimizer.iter_component(
        ParaStates(func=HyperGP.ops.RandTrCrv(), source=["p_list", "p_list"], to=["p_list", "p_list"],
                    mask=[lambda x=int(pop_size / 2):random.sample(range(pop_size), x), lambda x=int(pop_size / 2):random.sample(range(pop_size), x)]),
        ParaStates(func=HyperGP.ops.RandTrMut(), source=["p_list", ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=pop_size), True], to=["p_list"],
                    mask=[random.sample(range(pop_size), pop_size), 1, 1]),
        ParaStates(func=HyperGP.executor, source=["p_list", "input", "pset"], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=evaluation, source=["output", "target"], to=["fit_list"],
                    mask=[1, 1]),
        ParaStates(func=selection, source=["p_list", "pp_list", "fit_list", "pfit_list"], to=["p_list", "pp_list", "fit_list", "pfit_list"],
                    mask=[1, 1, 1, 1])
    )
optimizer.monitor(HyperGP.monitors.statistics_record, "fit_list")
optimizer.run(50)