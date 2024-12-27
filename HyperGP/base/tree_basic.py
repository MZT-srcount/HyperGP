import collections
import copy

from .base_struct import States, BaseStruct


class Node(BaseStruct):  # [] copy必须init一个treenode才能更新visited从而更新祖上节点的缓存；如果Crossover采用迁移而非复制的方式，如何处理祖上节点缓存的更新

    def __init__(self, nodeval, states=None, module_states=None, **kwargs):
        self.nodeval = nodeval  # 该节点的值，Func则对应Func类；特征则对应int;常量则对应float
        # if states is not None or module_states is not None:
        super().__init__(states, module_states, **kwargs)

    @property
    def arity(self):
        return self.nodeval.arity

    @property
    def type(self):
        return type(self.nodeval)


    """"shallow_copy: only copy the current node """
    def shallowCopy(self):
        pass

    """"deep_copy: copy the whole tree in which the node is the root """
    def copy(self):
        return copy.deepcopy(self)

    def setVal(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'states':
                if not isinstance(value, dict):
                    raise ValueError('The value of states should be dict type')
                for key_s, value_s in value.items():
                    self.states[key_s] = value_s
                    # setattr(self.states, key_s, value_s)
            elif key == 'module_states':
                if not isinstance(value, dict):
                    raise ValueError('The value of states should be dict type')
                for key_s, value_s in value.items():
                    self.module_states[key_s] = value_s
                    # setattr(self.module_states, key_s, value_s)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError("TreeNode has no the attr: %s yet, please register it first", key)

    def __str__(self):
        return str(self.nodeval)

    def format(self, *args):
        if self.nodeval.arity > 0:
            _args = ", ".join(map("{{{0}}}".format, range(self.nodeval.arity)))
            seq = "{name}({args})".format(name=self.nodeval.name, args=_args)
            return seq.format(*args)
        else:
            return self.nodeval.name




# class Node(BaseStruct):  # [] copy必须init一个treenode才能更新visited从而更新祖上节点的缓存；如果Crossover采用迁移而非复制的方式，如何处理祖上节点缓存的更新
#
#     def __init__(self, nodeval, states=None, module_states=None, **kwargs):
#         self.nodeval = nodeval  # 该节点的值，Func则对应Func类；特征则对应int;常量则对应float
#         self.parent, self.childs = None, None
#         super().__init__(states, module_states, **kwargs)
#
#     @property
#     def arity(self):
#         return self.nodeval.arity
#
#     @property
#     def type(self):
#         return type(self.nodeval)
#
#     def setChilds(self, childs):
#         if len(childs) != self.nodeval.arity:
#             raise ValueError("child size %d not equal to the parent arity %d" % (len(childs), self.nodeval.arity))
#         self.childs = childs
#         for i in range(len(childs)):
#             self.childs[i].parent = (self, i)
#
#     @property
#     def isRoot(self):
#         if self.parent is None:
#             return True
#         return False
#
#     """"shallow_copy: only copy the current node """
#     def shallowCopy(self):
#         pass
#
#     """"deep_copy: copy the whole tree in which the node is the root """
#     def copy(self):
#         return copy.deepcopy(self)
#
#     def setVal(self, **kwargs):
#         for key, value in kwargs.items():
#             if key == 'states':
#                 if not isinstance(value, dict):
#                     raise ValueError('The value of states should be dict type')
#                 for key_s, value_s in value.items():
#                     self.states[key_s] = value_s
#                     # setattr(self.states, key_s, value_s)
#             elif key == 'module_states':
#                 if not isinstance(value, dict):
#                     raise ValueError('The value of states should be dict type')
#                 for key_s, value_s in value.items():
#                     self.module_states[key_s] = value_s
#                     # setattr(self.module_states, key_s, value_s)
#             elif hasattr(self, key):
#                 setattr(self, key, value)
#             else:
#                 raise ValueError("TreeNode has no the attr: %s yet, please register it first", key)
#
#     def __str__(self):
#         return str(self.nodeval)
#
#     def format(self, *args):
#         if self.nodeval.arity > 0:
#             _args = ", ".join(map("{{{0}}}".format, range(self.nodeval.arity)))
#             seq = "{name}({args})".format(name=self.nodeval.name, args=_args)
#             return seq.format(*args)
#         else:
#             return self.nodeval.name
#
#     def traversal(self, func=None, **kwargs):
#         stack = [self]
#         tstack = []
#         while stack:
#             node = stack.pop()
#             tstack.append(node)
#             if node.childs is not None:
#                 stack.extend(list(map(lambda i: node.childs[i], range(len(node.childs) - 1, -1, -1))))
#             if func is not None:
#                 func(**kwargs)
#         return tstack
#
#
#     @property
#     def list(self):
#         return self.traversal()
#
#
#     def __getitem__(self, item):
#
#         node_list = self.list
#         if isinstance(item, slice):
#             try:
#                 return node_list[item.start:item.stop]
#             except ValueError:
#                 raise ValueError("input item out of range (0, %d)" % len(node_list))
#         if not isinstance(item, int):
#             raise ValueError("input should be 'int' type, instead of '%s'" % type(item))
#         if len(node_list) <= item:
#             raise ValueError("index '%d' should no longer than tree size '%d'" % (item, len(node_list)))
#
#         return node_list[item]
#
#
#     def replace(self, tr):
#         if self.arity != tr.arity:
#             raise ValueError("The arity of two node should keep same, while '%d' != '%d'"%(self.arity, tr.arity))
#         tr.parent = self.parent
#         tr.childs = self.childs
#         self.parent[0].childs[self.parent[1]] = tr
#         for i in range(len(self.childs)):
#             self.childs[i].parent[0] = tr
#
#     def trReplace(self, tr):
#         old_tr_parent = tr.parent
#         tr.parent = self.parent
#         if not self.isRoot:
#             self.parent[0].childs[self.parent[1]] = tr
#         return old_tr_parent



"""=========================Test part=========================="""
if __name__ == '__main__':
    t = Node(10)
    t1 = Node(10)
    t2 = Node(10)
    t.setVal(childs=[t1, t2])

    size = [0]
    def compute(size):
        size[0] += 1
    t.traversal(func=compute, size=size)
    print(size)

