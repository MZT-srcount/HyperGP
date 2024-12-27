from HyperGP.workflow import GpOptimizer
import numpy as np
import random
from HyperGP.library.population import Population, ProgBuildStates
from HyperGP.operators.crossover.tree_crv import RandTrCrv
from HyperGP.operators.mutation.tree_mut import RandTrMut
from HyperGP.library.primitive_set import PrimitiveSet
from HyperGP.base.base_struct import States
from HyperGP import executor, Tensor
import HyperGP
from HyperGP.monitors import statistics_record
from HyperGP import library, nn

def set_rdmask(size):
    return [random.randint(0, size - 1) for i in range(size)]

def set_prmask(size):
    cdd = random.sample(range(size), size)
    return [[cdd[i] for i in range(0, size, 2)], [cdd[i] for i in range(1, size, 2)]]

def set_armask(size):
    return [i for i in range(size)]

def exec_preprocess(states):
    return states[0]


def shuffle(progs):
    sample_list = random.sample(progs, len(progs))
    print("sample_list: ", len(sample_list))
    return sample_list

import time

def run():
    def evaluation(output, output_tensor):
        r1 = HyperGP.tensor.sub(output, output_tensor, dim_0=1)
        # r1 = output.substract(output_tensor, dim=1)
        return (r1 * r1).sum(dim=1).sqrt().numpy()
    
    
    pop_size = 1000
    pset = PrimitiveSet(input_arity=1,  primitive_set=[('add', HyperGP.tensor.add, 2),('sub', HyperGP.tensor.sub, 2)])
    pop = Population(parallel=False)
    pop.initPop(pop_size=pop_size, prog_paras=ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=100))
    pop.stateRegister(cprogs = pop.states['progs'].copy)
    input = np.random.uniform(0, 10, size=(1, 10000))
    
    prog_list = [States(prog_1=pop.states['cprogs'].indivs[i], prog_2=pop.states['cprogs'].indivs[i + 1]) for i in range(0, len(pop.states['cprogs'].indivs), 2)]
    
    output_tensor = Tensor(np.random.uniform(0, 10, size=(10000)))

    print(".........optimizer test begin.........")
    optimizer = GpOptimizer()

    optimizer.status_init(
        p_list=pop.states['cprogs'].indivs,
        fit_list = pop.states['cprogs'].fitness,
        input=Tensor(input),
        pset=pset,
        output=None
    )

    # optimizer.monitor(tool=, object=, save_path=)
    from HyperGP.library.states import ParaStates

    optimizer.iter_component(
        ParaStates(func=RandTrCrv(), source=[optimizer.p_list, optimizer.p_list], to=[optimizer.p_list, optimizer.p_list],
                    mask=set_prmask(pop_size)),
        ParaStates(func=shuffle, source=[optimizer.p_list], to=[optimizer.p_list],
                    mask=[1]),
        ParaStates(func=RandTrMut(), source=[optimizer.p_list, ProgBuildStates(pset=pset, depth_rg=[2, 3], len_limit=100), True], to=[optimizer.p_list],
                    mask=[set_armask(pop_size), 1, 1]),
        ParaStates(func=executor, source=[optimizer.p_list, "input", pset], to=["output", None],
                    mask=[1, 1, 1]),
        ParaStates(func=evaluation, source=["output", output_tensor], to=[optimizer.fit_list],
                    mask=[1, 1])
    )
    optimizer.monitor(tool=statistics_record, track_object='fit_list', save_path='./res.txt')
    optimizers = [optimizer]
    # for i in range(5):
    #     optimizers.append(optimizer.detach())


    st = time.time()
    for optimizer in optimizers:
        optimizer.run(50)
        # new_optimizer.run(10, async_parallel=True)
    for optimizer in optimizers:
        optimizer.wait()
    # new_optimizer.wait()
    print('finish_time: ', time.time() - st)

if __name__ == "__main__":
    
    # import multiprocessing
    # multiprocessing.freeze_support()
    run()