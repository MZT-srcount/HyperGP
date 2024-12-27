from .base_struct import States, BaseStruct
import copy


class Program(BaseStruct):
    def __init__(self, state=None, module_states=None, **kwargs):
        super().__init__(state, module_states, **kwargs)
        self.stateRegister(record=[])
        self.stateRegister(cash_record=[])
        self.encode = None

    def depth(self):
        encode_list = self.list()
        stack = [0]
        max_depth = 0
        for elem in encode_list:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth
    
    def buildProgram(self, cond, **kwargs):
        raise NotImplementedError("Function 'buildProgram' details not provided")
    
    # def __getstate__(self):
    #     return (self.encode, self.states)
    
    # def __setstate__(self, states):
    #     self.encode = states[0]
    #     self.states = states[1]

    def make(self, encode, states, memo):
        self.encode = copy.copy(encode)
        self.states = copy.deepcopy(states, memo)
        return self


    def copy(self):
        return copy.deepcopy(self)
    #
    # def slice(self, begin=None, end=None):
    #     return self.root.slice(begin, end)

