import copy

from HyperGP.base.prog_basic import Program
from ..states import ProgBuildStates
from ..utils import HalfAndHalf


class GEPIndv(Program):
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
    def buildProgram(self, cond: ProgBuildStates, method, node_states=None):
        encode = method(cond, node_states)
        # self.stateRegister(encode=root)
        self.encode = encode

    def __len__(self):
        return len(self.encode)

    def list(self, parent=False, childs=False):
        """transform the gep-encode to the preorder list"""

        pc_list = []

        '''child list'''
        c_list = []
        # child_idx = [1]
        # child_num = self.encode[0].arity
        # def add_idx(i):
        #     child_idx[0] += self.encode[i].arity
        #     return child_idx[0]
        # child_idx_list = [1] + [add_idx(i) for i, node in enumerate(self.encode) if i > 0]
        # c_list = [[child_idx_list[i] + z for z in range(self.encode[i].arity)] if self.encode[i].arity > 0 else [] for i, node in enumerate(self.encode)]
        
        child_idx = 1
        child_num = self.encode[0].arity
        for i, node in enumerate(self.encode):
            if child_num > 0:
                c_list.append([child_idx + z for z in range(child_num)])
            else:
                c_list.append([])
            child_idx += child_num
            child_num = self.encode[i + 1 if i + 1 < len(self.encode) else i].arity

        if childs:
            pc_list.append(c_list)
        
        '''parent list'''
        if parent:
            p_list = []
            def set_parent(child_list, p_idx):
                for node in enumerate(child_list):
                    p_list[node] = [p_idx]
            for i, nodes in enumerate(c_list):
                if len(nodes) > 0:
                    set_parent(nodes, i)
            pc_list.append(p_list)

            assert 0==1, "the implementation should be finished first"
        
        '''preorder list'''
        pre_list = []
        queue_obj = []
        cur_idx = 0
        queue_obj.append(cur_idx)
        while len(queue_obj) > 0:
            c_idx = queue_obj.pop()
            if c_idx >= len(self.encode):
                assert 0==1, "{S}, {S2}, {S1}".format(S=self.__str__(), S2=str(self), S1=[str(s) for s in self.encode])
            pre_list.append(self.encode[c_idx])
            for node in range(len(c_list[c_idx]) - 1 , -1, -1):
                queue_obj.append(c_list[c_idx][node])
        
        if not parent and not childs:
            # print('pre_list: ', pre_list)
            return pre_list
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
        self.encode.__setitem__(key, value)

    def copy(self):
        return copy.deepcopy(self, {})

    def __str__(self):
        def format(node, *args):
            if node.arity > 0:
                _args = ", ".join(map("{{{0}}}".format, range(node.arity)))
                seq = "{name}({args})".format(name=str(node), args=_args)
                return seq.format(*args)
            else:
                return str(node)
        
        string = ""
        stack = []
        arity, parent_idx = 0, -1
        parent_list = [-1]
        finish = False
        if self.encode[0].arity == 0:
            return str(self.encode[0])
        # print("self.encode: ", [str(s) for s in self.encode])
        # print("========================")
        for i, node in enumerate(self.encode):
            stack.append((i, []))
            '''find leave node'''
            if node.arity == 0 and i != 0:
                stack[parent_list[i]][-1].append(str(node))

                p_id = parent_list[i]
                while len(stack[p_id][1]) == self.encode[p_id].arity:
                    parent = parent_list[p_id]
                    prim, args = self.encode[p_id], stack[p_id][1]
                    string = format(prim, *args)
                    if parent == -1:
                        finish=True
                        break
                    stack[parent][1].append(string)
                    p_id = parent
            
            if finish:
                break
            '''record parent'''
            while arity == 0:
                parent_idx += 1
                if parent_idx > i:
                    print(parent_idx, i, len(self.encode), [str(s) for s in self.encode])
                    assert 0==1
                    break
                arity = self.encode[parent_idx].arity
            parent_list.append(parent_idx)

            arity -= 1
        assert len(stack[0][1]) == self.encode[0].arity, "can not transform the ind into str..."
        return string





