
class States(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            self[key] = value
    def __getattr__(self, item):
        if item in self:
            return self[item]
        
    def __setattr__(self, key, item):
            self[key] = item
    
    # def __str__(self):
    #     return str(self.__dict__)

class BaseStruct:
    available_mods = {}
    gmodule_states, gstates = States(), States()   #保护静态属性, 静态属性
    def __init__(self, states:dict=None, module_states:dict=None, **kwargs):
        '''initialize the states'''
        self.module_states = {}
        self.states = {}

        '''initialize states set'''
        if states is not None:
            for key, value in states.items():
                if callable(value):
                    self.states[key] = value()
                    # setattr(self.states, key, value())

        '''initialize module states set'''
        if module_states is not None:
            for key, value in module_states.items():
                if callable(value):
                    self.module_states[key] = value()
                    # setattr(self.module_states, key, value())

        '''initialize'''
        if states is not None:
            if not isinstance(states, dict):
                raise ValueError('The value of states should be dict type')
            for key_s, value_s in states.items():
                self.states[key_s] = value_s
                # setattr(self.states, key_s, value_s)
        if module_states is not None:
            if not isinstance(module_states, dict):
                raise ValueError('The value of states should be dict type')
            for key_s, value_s in module_states.items():
                self.module_states[key_s] = value_s
                # setattr(self.module_states, key_s, value_s)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def gstateRegister(**kwargs):
        for key, value in kwargs.items():
            BaseStruct.gstates[key] = value
            # setattr(BaseStruct.states, key, value)

    @staticmethod
    def gmoduleRegister(**kwargs):
        for key, value in kwargs.items():
            BaseStruct.gmodule_states[key] = value
            # setattr(BaseStruct.module_states, key, value)

    @staticmethod
    def gfuncRegister(func, *args, **kwargs):
        setattr(BaseStruct, func(*args, **kwargs))

    def funcRegister(self, func, *args, **kwargs):
        setattr(self, func(*args, **kwargs))

    def stateRegister(self, **kwargs):
        for key, value in kwargs.items():
            self.states[key] = value

    def moduleRegister(self, **kwargs):
        for key, value in kwargs.items():
            self.module_states[key] = value



"""=============================TEST==========================="""
if __name__ == '__main__':
    s = States()
    s['did'] = 100
    # setattr(s, 'did', 100)
    print(s.items())