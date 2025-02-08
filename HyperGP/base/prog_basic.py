from .base_struct import States, BaseStruct
from .func_basic import Constant
import copy, numpy as np
from numba import njit, jit


class Program(BaseStruct):
    def __init__(self, state=None, module_states=None, encode=None, **kwargs):
        super().__init__(state, module_states, **kwargs)
        if state is None:
            self.states['record'], self.states['cash_record'] = [], []

        self._pset_map = {}
        if encode is None:
            self._encode = None
            self._encode_array = None
        else:
            replace_array = np.array([elem for node in encode for elem in (node.idx if node.idx != -1 else node.val, node.arity, (0 if node.arity > 0 else 1) if node.idx != -1 else 2)])
            self._pset_map = {(node.idx if node.arity > 0 else -(node.idx + 1)):node for node in encode if node.idx != -1 and (node.idx if node.arity > 0 else -(node.idx + 1)) not in self._pset_map}
            self._encode_array = replace_array
            # self.stateRegister(encode=root)
            self._encode = encode

    @staticmethod
    @jit(nopython=True)
    def _depth(_encode_array):
        encode_list = _encode_array
        stack = [0]
        max_depth = 0
        for index in range(0, len(encode_list), 3):
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * encode_list[index + 1])
        return max_depth
    
    def depth(self):
        return self._depth(self._encode_array)

    def buildProgram(self, cond, method, node_states=None, **kwargs):
        
        encode = method(cond, node_states)
        replace_array = np.array([elem for node in encode for elem in (node.idx if node.idx != -1 else node.val, node.arity, (0 if node.arity > 0 else 1) if node.idx != -1 else 2)])

        self._pset_map = {(node.idx if node.arity > 0 else -(node.idx + 1)):node for node in encode if node.idx != -1 and (node.idx if node.arity > 0 else -(node.idx + 1)) not in self._pset_map}
        self._encode_array = replace_array
        # self.stateRegister(encode=root)
        self._encode = encode

        # raise NotImplementedError("Function 'buildProgram' details not provided")
    
    # def __getstate__(self):
    #     return (self.encode, self.states)
    
    # def __setstate__(self, states):
    #     self.encode = states[0]
    #     self.states = states[1]

    def make(self, encode_array, states, pset_map, memo):
        self._encode_array = encode_array.copy()
        self.states = copy.deepcopy(states, memo)
        self._pset_map = pset_map.copy()
        return self

    @staticmethod
    @jit(nopython=True)
    def _slice(_encode_array, begin=None, end=None):
         
        if begin is None:
            begin = 0
        encode_array = _encode_array 

        start_idx = begin * 3 + 1
        total = encode_array[start_idx]
        end = begin + 1
        current_idx = end * 3 + 1

        while total > 0:
            total += encode_array[current_idx] - 1
            end += 1
            current_idx += 3

        return begin, end
    
    def slice(self, begin=None, end=None):
        
        """
        Generate a slice object that defines the range of a subtree with the element of the 'begin' index as its root.
        If the 'begin' is None, then return the slice object with begin = 0

        Args:
            begin(int): determine the subtree slice range with which element as a root.
        
        Returns:
            Return a slice object representing the range of a subtree with given element of 'begin' index as root 

        Examples:
            >>> print(ind.slice(0))
            xxxxxx
            >>> print(ind.slice(2))
            xxxxxx
            
        """

        slice_be = (self._slice(self._encode_array, begin, end))
        return slice(slice_be[0], slice_be[1], 1)

    def get_encode(self, item):
        return (self._encode_array[item.start * 3:item.stop * 3], self._pset_map)

    def set_encode(self, key, value):
        if not (isinstance(key, slice) or isinstance(key, int)):
            raise ValueError("The key should be slice or int, but '{TYPE}' is given".format(TYPE=type(key)))
        
        encode_array = self._encode_array
        self._encode = None

        if self._encode_array is not None:
            key_slice = key if isinstance(key, slice) else slice((key, key + 1))
            elems = []
            replace_array = value[0]
            self._encode_array = np.concatenate((self._encode_array[:key_slice.start * 3], replace_array, self._encode_array[key_slice.stop * 3:]))
            self._pset_map.update(value[1])
        else:
            replace_array = value[0]
            self._encode_array = replace_array
            self._pset_map = value[1]
            
    def __getitem__(self, item):
        encode_array = self._encode_array
        if self._encode is None and encode_array is not None:
            self._encode = [self._pset_map[int(encode_array[index] if encode_array[index + 1] > 0 else -(encode_array[index] + 1))] if encode_array[index + 2] != 2 else Constant(encode_array[index]) for index in range(0, len(encode_array), 3)]
        return self._encode[item]

    def __setitem__(self, key, value):
        if not (isinstance(key, slice) or isinstance(key, int)):
            raise ValueError("The key should be slice or int, but '{TYPE}' is given".format(TYPE=type(key)))
        
        encode_array = self._encode_array
        if self._encode is not None:
            self._encode.__setitem__(key, value)
        # if self._encode is None and encode_array is not None:
        #     pass
        #     # self._encode = [self._pset_map[int(encode_array[index] if encode_array[index + 1] > 0 else -(encode_array[index] + 1))] if encode_array[index + 2] != 2 else Constant(encode_array[index]) for index in range(0, len(encode_array), 3)]
        # else:
        #     self._encode.__setitem__(key, value)

        if self._encode_array is not None:
            key_slice = key if isinstance(key, slice) else slice((key, key + 1))
            elems = []
            replace_array = np.array([elem for node in value for elem in self.gen_array(node)])
            self._encode_array = np.concatenate((self._encode_array[:key_slice.start * 3], replace_array, self._encode_array[key_slice.stop * 3:]))
            # self._pset_map.update({int((replace_array[index] if replace_array[index + 1] > 0 else -(replace_array[index] + 1))):value[int(index / 3)] for index in range(0, len(replace_array), 3) if replace_array[index + 2] != 2 and int(replace_array[index] if replace_array[index + 1] > 0 else -(replace_array[index] + 1)) not in self._pset_map})
        else:
            replace_array = np.array([elem for node in value for elem in (node.idx if node.idx != -1 else node.val, node.arity, (0 if node.arity > 0 else 1) if node.idx != -1 else 2)])
            self._encode_array = replace_array
            self._pset_map = {int((replace_array[index] if replace_array[index + 1] > 0 else -(replace_array[index] + 1)) * replace_array[index]):value[int(index / 3)] for index in range(0, len(replace_array), 3) if replace_array[index + 2] != 2 and int(replace_array[index] if replace_array[index + 1] > 0 else -(replace_array[index] + 1)) not in self._pset_map}
            
    def deepcopy(self):
        return copy.deepcopy(self)

    def update(self, target_ind):
        self._encode_array, self._pset_map = target_ind._encode_array.copy(), target_ind._pset_map
        return self
        # return {'_encode_array':self._encode_array.copy(), '_pset_map':self._pset_map}
        

    def gen_array(self, node):
        idx, arity = node.idx, node.arity
        if idx != -1 and int(idx if arity > 0 else -(idx + 1)) not in self._pset_map:
            self._pset_map.update({int(idx if arity > 0 else -(idx + 1)):node})
        return (idx if idx != -1 else node.val, arity, (0 if arity > 0 else 1) if idx != -1 else 2)

    
    #
    # def slice(self, begin=None, end=None):
    #     return self.root.slice(begin, end)

