import warnings

import numpy as np
from HyperGP.base.func_basic import Constant
import numpy as np
from HyperGP.base.base_struct import States
from ..src import pygp_cash


class CashManager:

    def __init__(self, limit=10000):
        # super.__init__()
        self.cash = pygp_cash.CashList(limit)

    def add(self):
        pass

    def __getitem__(self, item):
        return self.cash.get(item)

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        return self.cash.insert(key, value, value.shape)

    def getSemantic(self, item):
        if self.cash.findElem(item):
            smt = self.cash.get(item)
            self.cash.reinsert(item, 0)
        return smt

    def set(self, keys, items, **kwargs):
        """set cash with semantic"""
        for i, key in enumerate(keys):
            if not self.cash.findElem(key):
                self[key] = value
        for key, value in kwargs.items():
            if not self.cash.findElem(key):
                self[key] = value


    """
    generate a new ind list, with a cash list that replace the origin node.
    to gen with just one time traversal, use two list, one is the ind, the other one is generated for the record node.
    """
    def getCash(self, ind):

        nodes_arity, nodes_str = [], []
        prog_list = ind.list()
        nodes_arity = [node.arity for node in prog_list]
        nodes_str = [str(node.nodeval) for node in prog_list]
        cash_records = []
        for i, node in enumerate(prog_list):
            if ('cash_record' in node.states and node.states['cash_record']) or ('record' in node.states and node.states['record']):
                cash_records.append(i)
                node.states['cash_record'] = False
        '''To get cash list and the node with record point'''   
        (cash_arrays, cash_list, sym_set, ind_after_cash) = pygp_cash.getCash(self.cash, nodes_arity, nodes_str, cash_records)
        return ind_after_cash, States(cash_arrays=cash_arrays, cash_list=cash_list, sym_set=sym_set, nodes_arity=nodes_arity, nodes_str=nodes_str, cash_records=cash_records)

    """
    generate a new ind list, with a cash list that replace the origin node.
    to gen with just one time traversal, use two list, one is the ind, the other one is generated for the record node.
    """
    def getCashes(self, inds, pset):

        nodes_arity, nodes_str, cash_records, idxs, constant_set = [], [], [], [], []
        import time
        st = time.time()
        f_nvec = pset.primitiveSet + pset.terminalSet
        f_avec = [pset.genFunc(f_str).arity for f_str in pset.primitiveSet] + [pset.genTerminal(t_str).arity for t_str in pset.terminalSet]
        p_len = len(pset.primitiveSet)
        cash_records = [ind.states['cash_record'] + ind.states['record'] for ind in inds]
        progs_list = [ind.list() for ind in inds]
        def const_set(node):
            constant_set.append(node.val if isinstance(node, Constant) else node.func())
            return -len(constant_set)
        idxs = [[((node.idx + p_len) if node.idx != -1 else const_set(node)) if node.arity == 0 else node.idx for node in prog] for prog in progs_list]

        print(time.time() - st)
        st = time.time()
        '''To get cash list and the node with record point'''   
        (cash_arrays, cash_list, sym_set, ind_after_cash) = pygp_cash.getCashes(self.cash, (f_nvec, f_avec, p_len), idxs, constant_set, cash_records)
        
        print(time.time() - st)
        return ind_after_cash, States(cash_arrays=cash_arrays, cash_list=cash_list, sym_set=sym_set, f_attrs=(f_nvec, f_avec, p_len), idxs=idxs, cash_records=cash_records)


if __name__ == '__main__':
    a, b = 1, [1, 2, 3]
    print(b, a, b)