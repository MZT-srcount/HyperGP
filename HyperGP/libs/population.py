import numpy as np

from HyperGP.base.pop_basic import PopBase
from HyperGP.libs.states import ProgBuildStates
from HyperGP.libs.representation.TGP import TGPIndv
from HyperGP.libs.utils import ProgBuildMethod
from HyperGP.base.base_struct import States
from HyperGP.operators.crossover.tree_crv import CrossoverMethod
from HyperGP.operators.mutation.tree_mut import MutMethod
from HyperGP.operators.evaluation.evaluate import EvaluateMethod


def build(prog_para, prog_states, node_state, method=None, indiv_type=TGPIndv):
    # print('prog_para: ', str(prog_states))
    indiv = indiv_type(states=prog_states)
    if method is not None:
        indiv.buildProgram(prog_para, method=method, node_states=node_state)
    else:
        indiv.buildProgram(prog_para, node_states=node_state)

    return indiv

class Population(PopBase):
    def __init__(self, parallel=False, GPU=False, states=None, module_states=None, **kwargs):
        super().__init__(parallel, GPU, states, module_states, **kwargs)

    def initPop(self, pop_size, prog_paras: ProgBuildStates or [ProgBuildStates], prog_states: dict or list = None, node_states=None, methods:ProgBuildMethod or list=None, indivs_type=TGPIndv, **kwargs):
        prog = []
        fitness = []

        if isinstance(prog_paras, list):
            if len(prog_paras) != pop_size:
                raise ValueError('prog_paras should equal to pop_size as a list')
        else:
            prog_paras = [prog_paras] * pop_size

        if isinstance(prog_states, list):
            if len(prog_states) != pop_size:
                raise ValueError('len(prog_states) should equal to pop_size as a list')
        else:
            prog_states = [prog_states] * pop_size

        if methods is not None and isinstance(methods, list):
            if len(methods) != pop_size:
                raise ValueError('len(prog_states) should equal to pop_size as a list')
        elif not isinstance(methods, list):
            methods = [methods] * pop_size

        if indivs_type is not None and isinstance(indivs_type, list):
            if len(indivs_type) != pop_size:
                raise ValueError('len(prog_states) should equal to pop_size as a list')
        elif not isinstance(indivs_type, list):
            indivs_type = [indivs_type] * pop_size

        if isinstance(node_states, list):
            if len(node_states) != pop_size:
                raise ValueError('len(prog_states) should equal to pop_size as a list')
        else:
            node_states = [node_states] * pop_size
        para = [States(prog_para=prog_paras[ind], prog_states=prog_states[ind], node_state=node_states[ind], method=methods[ind], indiv_type=indivs_type[ind]) for ind in range(pop_size)]
        # para = list(zip(prog_paras, node_states, methods))
        # print(str(para[0]))
        # print(para[0].prog_para)
        # assert 0==1

        if 'parallel' in self.module_states:
            self.states['progs'].indivs = self.parallel(build, para)
        else:
            self.states['progs'].indivs = list(map(lambda prog_id: build(**(para[prog_id])), list(range(pop_size))))

        self.states['progs'].fitness = [] * pop_size

    def crossover(self, method: CrossoverMethod or list, states: States or list = None, parallel=False, **kwargs):
        #new_states, old_states
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            for i, state in enumerate(states):
                ret_cond.append(method(**state, **kwargs)
                             if not isinstance(method, list)
                             else method[i](**state, **kwargs))
        return ret_cond

    def mutation(self, method: MutMethod or list, states: States or list = None, parallel=False, **kwargs):
        #new_states, old_states
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            for i, state in enumerate(states):
                ret_cond.append(method(**state, **kwargs)
                             if not isinstance(method, list)
                             else method[i](**state, **kwargs))
        return ret_cond

    def evaluation(self, method: EvaluateMethod or list, states: States or list = None, parallel=False, **kwargs):
        #new_states, old_states
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            for i, state in enumerate(states):
                ret_cond.append(method(**state, **kwargs)
                             if not isinstance(method, list)
                             else method[i](**state, **kwargs))
        return ret_cond

    def execution(self, method: EvaluateMethod or list, states: States or list = None, parallel=False, **kwargs):
        #new_states, old_states
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            for i, state in enumerate(states):
                ret_cond.append(method(**state, **kwargs)
                             if not isinstance(method, list)
                             else method[i](**state, **kwargs))
        return ret_cond

    def selection(self, method, states: States or list = None, parallel=False, **kwargs):
        #new_states, old_states
        if isinstance(method, list) and len(method) != len(states):
            raise ValueError('The method size %d not equal to the cond size %d' % (len(method), len(states)))
        if 'parallel' in self.gmodule_states and parallel:
            ret_cond = self.parallel(method, states, **kwargs)
        else:
            ret_cond = []
            for i, state in enumerate(states):
                ret_cond.append(method(**state, **kwargs)
                             if not isinstance(method, list)
                             else method[i](**state, **kwargs))
        return ret_cond

    def cash_upadte(self):
        raise NotImplementedError('Not finish yet...')

    def rewrite(self, func, *args, **kwargs):
        pass

    def iter(self, iteration, cond):
        pass


"""=============================TEST==========================="""
if __name__ == '__main__':
    import math
    from primitive_set import PrimitiveSet

    def add(a, b):
        return a+b
    def c(**kwargs):
        return add(**kwargs)

    States(a = 10, b=100)
    print(c(**States(a = 10, b=100)))
    # pop = Population()
    # pop.initPop(pop_size=100)
    # iteration = 100
    # for iter in range(iteration):
    #     pop.variation()
    #     pop.execution()
    #     pop.evaluation()
    #     pop.selection()
    def add(a, b):
        return a + b
    primitive_set = [('pow', math.pow, 2), ('add', add, 2)]
    pset = PrimitiveSet(input_arity=10, primitive_set=primitive_set)
    pop = Population()
    pstates = ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=200)
    pop.initPop(pop_size=100, prog_paras=pstates)