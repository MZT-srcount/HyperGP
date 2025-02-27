import copy
import random
import numpy as np

from HyperGP.base.prog_basic import Program
from ..states import ProgBuildStates
from ..utils import HalfAndHalf

prog_basic_list = np.arange(10000, dtype=np.int32)

class TGPIndv(Program):
    """
    We provide the ``TGPIndv`` class to build the tree structure program

    Note:
        The encode list is a collection of pset elements without deep copy.

    """
    def __init__(self, states=None, encode=None, module_states=None, **kwargs):
        """
        Initialize the program

        Args:
            states(HyperGP.States): the states assign to a program.
            encode: generate a new ``TGPIndv`` with a given encode.
            kwargs: the attrs assign to a program.
        
        Returns:
            returns a new ``TGPIndv``

        Examples:
            >>> from HyperGP.representation import TGPIndv
            >>> from HyperGP.states import States
            >>> ind = TGPIndv()

            Initialize with states and attrs:
            >>> ind = TGPIndv(states=States(elim_prob=1, rk=0), win_num=0)
            >>> print(ind.states)
            xxxxx
            >>> print(TGPIndv.win_num)
            xxxxx

        """
        super().__init__(state=states, module_states=module_states, encode=encode, **kwargs)
        # if states is not None:
        #     if 'module_states' not in states and 'states' not in states:
        #         super().__init__(state=states, **kwargs)
        #     else:
        #         super().__init__(**states, **kwargs)
        # else:
        #     super().__init__(state=None, module_states=None, **kwargs)

        # if encode is not None:
        #     self._encode = encode

    """"""
    def buildProgram(self, cond: ProgBuildStates, method=HalfAndHalf(), node_states=None):
        
        """
        Build the program

        Args:
            cond(ProgBuildStates): The states needed to generate the program with given method, which will be used as uniform formal parameter of the generation method.
            method: the method to generate a encode list, with `cond` parameter as input.
            node_states: the states want .
        
        Returns:
            returns a new ``TGPIndv``

        Examples:
            >>> from HyperGP.states import ProgBuildStates
            >>> prog_states = ProgBuildStates(pset=pset, depth_rg=[2, 6], len_limit=100)
            >>> ind.build(prog_states)
            >>> print(ind)
            xxxxxxx
            
        """
         
        encode = method(cond, node_states)
        replace_array = np.array([elem for node in encode for elem in (node.idx if node.idx != -1 else node.val, node.arity, (0 if node.arity > 0 else 1) if node.idx != -1 else 2)], dtype=np.float32)

        self._pset_map = {(node.idx if node.arity > 0 else -(node.idx + 1)):node for node in encode if node.idx != -1 and (node.idx if node.arity > 0 else -(node.idx + 1)) not in self._pset_map}
        self._encode_array = replace_array
        # self.stateRegister(encode=root)
        self._encode = encode


    # def list(self, parent=False, childs=False):
    #     if not parent and not childs:
    #         return list(range(len(self._encode)))


    def list(self, parent=False, childs=False):
         
        """
        Generate the preorder traversal list of the program

        Args:
            parent(bool): whether generate the parent list with the preorder traversal list.
            childs(bool): whether generate the child list with the preorder traversal list.
        
        Returns:
            Return the preorder traversal list.
            If parent or childs is true, then return a list: [preorder list, parent list if parent=True, child list if childs=True]

        Examples:
            >>> print(ind.list())
            xxxxxx
            >>> print(ind.list(child=True))
            xxxxxx
            >>> print(ind.list(parent=True, child=True))
            xxxxxx
            
        """

        if not parent and not childs:
            return prog_basic_list[:int(self._encode_array.shape[0] / 3)]
        pc_list = []
        if parent:
            p_list = [[] for z in range(int(self._encode_array.shape[0] / 3))]
            cur_arity = [[0, self._encode_array[1]]]
            for i in range(0, len(self._encode_array[1:]), 3):
                i = int(i / 3 + 1)
                arity = self._encode_array[1 + i * 3]
                idx, _ = cur_arity[-1]
                cur_arity[-1][1] -= 1
                p_list[i].append(idx)
                if cur_arity[-1][1] == 0:
                    cur_arity.pop()
                if arity > 0:
                    cur_arity.append([i, arity])
            pc_list.append(p_list)
        if childs:
            c_list = [[] for z in range(int(self._encode_array.shape[0] / 3))]
            cur_arity = [[0, self._encode_array[1].arity]]
            for i in range(0, len(self._encode_array[1:]), 3):
                i = int(i / 3 + 1)
                arity = self._encode_array[1 + i * 3]
                idx, _ = cur_arity[-1]
                cur_arity[-1][1] -= 1
                c_list[idx].append(i)
                if cur_arity[-1][1] == 0:
                    cur_arity.pop()
                if arity > 0:
                    cur_arity.append([i, arity])
            pc_list.append(c_list)
            if len(cur_arity) != 0:
                raise ValueError("Something wrong when search childs in list()")
            # print(len(c_list), len(c_list[0]), len(c_list[1]), c_list[0], c_list[1])
        return pc_list

    # def __getitem__(self, item):
    #     # return [ for i in self.encode[item]]
    #     return self.encode[item]

    # def __setitem__(self, key, value):

    #     if isinstance(key, slice):
    #         if key.start >= len(self):
    #             raise IndexError("Invalid slice object (try to assign a %s"
    #                              " in a tree of size %d). Even if this is allowed by the"
    #                              " list object slice setter, this should not be done in"
    #                              " the PrimitiveTree context, as this may lead to an"
    #                              " unpredictable behavior for searchSubtree or evaluate."
    #                              % (key, len(self)))
    #         total = value[0].arity
    #         for node in value[1:]:
    #             total += node.arity - 1
    #         if total != 0:
    #             raise ValueError("Invalid slice assignation : insertion of"
    #                              " an incomplete subtree is not allowed in PrimitiveTree."
    #                              " A tree is defined as incomplete when some nodes cannot"
    #                              " be mapped to any position in the tree, considering the"
    #                              " primitives' arity. For instance, the tree [sub, 4, 5,"
    #                              " 6] is incomplete if the arity of sub is 2, because it"
    #                              " would produce an orphan node (the 6).")
    #     elif value.arity != self[key].arity:
    #         raise ValueError("Invalid node replacement with a node of a"
    #                          " different arity.")
    #     self.encode.__setitem__(key, value)

    def __deepcopy__(self, memo):
        new_ind = TGPIndv()
        return new_ind.make(self._encode_array, self.states, self._pset_map, memo)
    
    def copy(self):
        """
        Returns a new ``TGPIndv`` with the same encode list and states.
        """
        new_ind = TGPIndv.__new__(TGPIndv)
        # new_ind = TGPIndv()
        new_ind.update(self)
        # new_ind._encode_array, new_ind._pset_map = self._encode_array.copy(), self._pset_map
        return new_ind

    def __str__(self):
        def format(node, *args):
            if node.arity > 0:
                _args = ", ".join(map("{{{0}}}".format, range(node.arity)))
                seq = "{name}({args})".format(name=node.name, args=_args)
                return seq.format(*args)
            else:
                return str(node)
        
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = format(prim, *args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)

        return string
        #
        # stack = [self.encode]
        # tstack = []
        # while stack:
        #     node = stack.pop()
        #     tstack.append((node, []))
        #     if node.childs is not None:
        #         stack.extend(node.childs)
        #     while tstack[-1][0].arity == len(tstack[-1][1]):
        #         f_node = tstack.pop()
        #         if len(tstack) == 0:
        #             return f_node[0].format(*f_node[1])
        #         tstack[-1][1].append(f_node[0].format(*f_node[1]))





