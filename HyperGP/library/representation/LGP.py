import copy
import random

from HyperGP.base.prog_basic import Program
from ..states import ProgBuildStates
from ..utils import HalfAndHalf

class Operator:
    def __init__(self, inputs, operator):
        self.inputs = inputs
        self.operator = operator
    
    def __str__(self):
        raise NotImplementedError("Operator unit is not implemented now")

class LGPIndv(Program):
    def __init__(self, states=None, encode=None, **kwargs):

        if states is not None:
            if 'module_states' not in states and 'states' not in states:
                super().__init__(state=states, **kwargs)
            else:
                super().__init__(**states, **kwargs)
        else:
            super().__init__(state=None, module_states=None, **kwargs)

        if encode is not None:
            self.encode = encode

    """"""
    def buildProgram(self, cond: ProgBuildStates, method=HalfAndHalf(), node_states=None):
        encode = method(cond, node_states)
        # self.stateRegister(encode=root)
        self.encode = encode

    def __len__(self):
        return len(self.encode)

    def list(self, parent=False, childs=False):
        pc_list = []
        if not parent and not childs:
            pre_list = []
            stack = [self.encode[0]]
            while len(stack) > 0:
                instruction = stack.pop()
                pre_list.append(instruction.operator)
                for input in instruction.inputs:
                    if isinstance(input, int):
                        assert input < len(self.encode), "the input id %d in instruction out of range, which should be smaller than %d" % (input, len(self.encode))
                        stack.append(self.encode[input])
                    else:
                        pre_list.append(input)
        if parent:
            p_list = [[] for z in range(len(pre_list))]
            cur_arity = [[0, pre_list[0].arity]]
            for i, node in enumerate(pre_list[1:]):
                i = i + 1
                idx, _ = cur_arity[-1]
                cur_arity[-1][1] -= 1
                p_list[i].append(idx)
                if cur_arity[-1][1] == 0:
                    cur_arity.pop()
                if node.arity > 0:
                    cur_arity.append([i, node.arity])
            pc_list.append(p_list)
            assert 0==1, "the implementation should be finished first"
        if childs:
            c_list = [[] for z in range(len(pre_list))]
            cur_arity = [[0, pre_list[0].arity]]
            for i, node in enumerate(pre_list[1:]):
                i = i + 1
                idx, _ = cur_arity[-1]
                cur_arity[-1][1] -= 1
                c_list[idx].append(i)
                if cur_arity[-1][1] == 0:
                    cur_arity.pop()
                if node.arity > 0:
                    cur_arity.append([i, node.arity])
            pc_list.append(c_list)
            assert len(cur_arity) == 0
        return pc_list



    def slice(self, begin=None, end=None):
        if begin is None:
            begin = 0
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def __getitem__(self, item):
        return self.encode[item]

    def __setitem__(self, key, value):

        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = value[0].arity
            for node in value[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5,"
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif value.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")
        self.encode.__setitem__(key, value)

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        def format(node, *args):
            if node.arity > 0:
                _args = ", ".join(map("{{{0}}}".format, range(node.arity)))
                seq = "{name}({args})".format(name=node.name, args=_args)
                return seq.format(*args)
            else:
                return node.name
        
        string = ""
        
        stack = [self.encode[0]]
        while len(stack) > 0:
            instruction = stack.pop()
            input_list = ["Reg[%d]"%(input) if isinstance(input, int) else str(input) for input in instruction.inputs]
            string = format(instruction.operator, *input_list)

            for input in instruction.inputs:
                if isinstance(input, int):
                    assert input < len(self.encode), "the input id %d in instruction out of range, which should be smaller than %d" % (input, len(self.encode))
                    stack.append(self.encode[input])

        return string

