import sys

import HyperGP.mods
from HyperGP.base.base_struct import BaseStruct
from HyperGP.library.states import PopBaseStates
import inspect
from HyperGP.mods import AvailableMods, __Mods

class PopBase(BaseStruct, __Mods):

    available_mods = AvailableMods()

    def __init__(self, parallel=False, GPU=False, states=None, module_states=None, **kwargs):
        from HyperGP.mods import ModBase
        super().__init__(states, module_states, **kwargs)

        # '''load the available module'''
        # for module_name, module in sys.modules.items():
        #     for name, obj in inspect.getmembers(module):
        #         if inspect.isclass(obj) and issubclass(obj, ModBase) and obj != ModBase:
        #             self.available_mods[name] = obj

        '''init state'''
        self.stateRegister(progs=PopBaseStates(), pprogs=PopBaseStates())
        print('self.states', self.states)
        # '''load the inner module'''
        if parallel:
            self.enable('parallel')
        # self.moduleRegister(parallel=self.__parallelEnable)
        # self.moduleRegister(GPU=self.__GPUEnable)


    """
    states:     [States x pop_size]
    prog_states:     [States x pop_size]
    node_states:     [States x pop_size]
    """
    def initPop(self, pop_size, prog_paras, prog_states=None, node_states=None, **kwargs):
        raise NotImplementedError("The implementation of function 'initPop' not provided")

    """
    mods: [module_1, ...], list the name of modules needed to enable
    """
    # def enable(self, mods):
    #     for module in mods:
    #         if module in self.available_mods and callable(self.available_mods[module]):
    #             self.available_mods[module](self)
    def enable(self, mod, **kwargs):
        if getattr(self, mod):
            self.__setattr__(mod, self.available_mods.__getattribute__(mod)())
            self.__getattribute__(mod)._popSet(self, **kwargs)


    def GPUMap(self, func, **kwargs):
        pass


    def __GPUEnable(self):
        pass



"""=============================TEST==========================="""
if __name__ == '__main__':
    p = PopBase(True)
    p.enable('parallel')
    p.states['progs'].fitness.extend([222, 111])
    p1 = PopBase(True)
    p1.enable('parallel')
    # print(p.states)
    p1.states['progs'].fitness.extend([111, 111])

    # print(p.available_mods)
    # print(p.module_states)
    print(p.states)
    print(p1.states)
    print(p.states['progs'])
    print(p1.states['progs'])
    # print(PopBaseStates())
    #p.module_states['parallel']()
    # setattr(s, 'did', 100)