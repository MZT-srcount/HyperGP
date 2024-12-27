from HyperGP.base.base_struct import States
from HyperGP.library.primitive_set import PrimitiveSet
import random
import copy

class WorkflowStates(States):
    def __init__(self):
        pass
    def __getattr__(self, item):
        if item not in self:
            return None
        return self[item]
    # def __deepcopy__(self, memo):
    #     return copy.deepcopy(self.__dict__, memo)

class ParaStates(States):
    def __init__(self, func, source, to, mask, parallel=False, func_mask=None):
        self["func"]=func
        self["source"]=source
        self["to"]=to
        self["mask"]=mask
        self["parallel"]=parallel
        self["func_mask"]=func_mask
    def __getattr__(self, item):
        if item not in self:
            self[item] = []
        return self[item]

class PopBuildStates(States):
    def __init__(self):
        self.pop_size: int = 0

class PopBaseStates:
    def __init__(self):
        self.indivs: list = []
        self.fitness: list = []
    def __str__(self):
        # print(self.prog)
        # print(self.fitness)
        return str([self.indivs, self.fitness])

    @property
    def copy(self):
        return copy.deepcopy(self)

class ProgBuildStates(States):
    # pset: PrimitiveSet
    # depth_rg: list
    # len_limit = None
    # rd_state = random

    def __init__(self, pset: PrimitiveSet, depth_rg:list, len_limit=None, **kwargs):#, rd_state=random):
        super().__init__()
        self.pset = pset
        self.depth_rg = depth_rg
        if len_limit is not None:
            self.len_limit = len_limit
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.rd_state = rd_state

class VarStates(States):

    def __init__(self, progs, fitness=None, **kwargs):
        from HyperGP.library.representation.TGP import TGPIndv
        self.progs: TGPIndv or [TGPIndv] = progs
        if fitness is not None:
            self.fitness = fitness
        self.__dict__.update(**kwargs)





"""=========================TEST========================"""
if __name__ == '__main__':
    from HyperGP.library.representation.TGP import TGPIndv
    # ps = ProgBuildStates()
    # print(ps.len_limit)
    var = VarStates(TGPIndv(), f=11, s='222')
    print(var.s)