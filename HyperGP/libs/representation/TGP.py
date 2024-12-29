import copy
import random

from HyperGP.base.prog_basic import Program
from ..states import ProgBuildStates
from ..utils import HalfAndHalf

class TGPIndv(Program):
    """
    We provide the ``TGPIndv`` class to build the tree structure program

    Note:
        The encode list is a collection of pset elements without deep copy.

    """
    def __init__(self, states=None, encode=None, **kwargs):
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
        # self.stateRegister(encode=root)
        self.encode = encode

    def __len__(self):
        return len(self.encode)

    def list(self, parent=False, childs=False):
         
        """
        Generate the postorder traversal list of the program

        Args:
            parent(bool): whether generate the parent list with the postorder traversal list.
            childs(bool): whether generate the child list with the postorder traversal list.
        
        Returns:
            Return the postorder traversal list.
            If parent or childs is true, then return a list: [postorder list, parent list if parent=True, child list if childs=True]

        Examples:
            >>> print(ind.list())
            xxxxxx
            >>> print(ind.list(child=True))
            xxxxxx
            >>> print(ind.list(parent=True, child=True))
            xxxxxx
            
        """

        if not parent and not childs:
            return self.encode
        pc_list = []
        if parent:
            p_list = [[] for z in range(len(self.encode))]
            cur_arity = [[0, self.encode[0].arity]]
            for i, node in enumerate(self.encode[1:]):
                i = i + 1
                idx, _ = cur_arity[-1]
                cur_arity[-1][1] -= 1
                p_list[i].append(idx)
                if cur_arity[-1][1] == 0:
                    cur_arity.pop()
                if node.arity > 0:
                    cur_arity.append([i, node.arity])
            pc_list.append(p_list)
        if childs:
            c_list = [[] for z in range(len(self.encode))]
            cur_arity = [[0, self.encode[0].arity]]
            for i, node in enumerate(self.encode[1:]):
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
            # print(len(c_list), len(c_list[0]), len(c_list[1]), c_list[0], c_list[1])
        return pc_list



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

        if begin is None:
            begin = 0
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    def __getitem__(self, item):
        # return [ for i in self.encode[item]]
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

    def __deepcopy__(self, memo):
        new_ind = TGPIndv()
        return new_ind.make(self.encode, self.states, memo)
    
    def copy(self):

        """
        Returns a new ``TGPIndv`` with the same encode list and states.
        """

        new_ind = TGPIndv()
        return new_ind.make(self.encode, self.states, {})


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





