from HyperGP.libs.representation.TGP import TGPIndv
from HyperGP.libs.primitive_set import PrimitiveSet
from HyperGP.base.func_basic import Constant
from HyperGP.base.base_struct import States
from .cash_manager import CashManager
import numpy as np
from ..src import pygp_utils
import itertools

"""[ ] TODO: Execution respond to get the output data of expression, there should be the ability to return the data need to record"""
class ExecutableGen:
    def __call__(self, progs:[TGPIndv], pset: PrimitiveSet, cash_manager = None):
        exec_len_max = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        id_init, id_p, id_max = len(pset.arguments) + len(progs), 0, 0
        init_output, output, constant_set, record_set, constant_strs_set = {}, {}, [], {}, []
        sym_set, exp_set, exp_iposi_set = [], [], []
        cash_list, cash_records = [], []
        cash_arrays = []
        nodes_arity, nodes_str, ind_after_cashes, idxs = [], [], [], []


        import time
        st = time.time()
        p_len = len(pset.primitiveSet)
        if cash_manager is not None:
            '''cash scan first, to find the subtree without cash'''
            ind_after_cashes, states = cash_manager.getCashes(progs, pset)
            cash_records, cash_arrays, cash_list, f_attrs, idxs, sym_set = states['cash_records'], states['cash_arrays'], states['cash_list'], states['f_attrs'], states['idxs'], states['sym_set']
            
        else:
            sym_set = 0
            f_nvec = pset.primitiveSet + pset.terminalSet
            f_avec = [pset.genFunc(f_str).arity for f_str in pset.primitiveSet] + [pset.genTerminal(t_str).arity for t_str in pset.terminalSet]
            f_attrs = (f_nvec, f_avec, p_len)
            def const_set(node):
                constant_set.append(node.val if isinstance(node, Constant) else node.func())
                constant_strs_set.append(str(node.val if isinstance(node, Constant) else node.func()))
                return -len(constant_set)
            post_list = [ind.list() for ind in progs]
            idxs = [[((node.idx + p_len) if node.idx != -1 else const_set(node)) if node.arity == 0 else node.idx for node in prog] for prog in post_list]

            ind_after_cashes = [[len(ind)] for ind in post_list]
            cash_records = [ind.states['cash_record'] + ind.states['record'] for ind in progs]
            
        if len(cash_arrays) > 0:
            cash_arrays = np.vstack(cash_arrays)
        else:
            cash_arrays = np.array(cash_arrays)
        # print('prepare: ', time.time() - st)
        st = time.time()
        exp_set, layer_info, records_posi, record_strs, x_len = pygp_utils.tree2graph(f_attrs,
                                          ind_after_cashes, idxs, sym_set, cash_list, cash_records, constant_strs_set, [len(pset.arguments), id_init, exec_len_max, p_len])
        # print('t2p: ', time.time() - st, len(exp_set) / 5, sum(len(ind) for ind in post_list))#, len(list(itertools.chain.from_iterable(idxs))) * exec_len_max)

        # print(sum([len(ind) for ind in ind_after_cashes]))
        # assert 0==1
        return exp_set, States(constants=constant_set, layer_info=layer_info, x_len=x_len, record_set=cash_records, records_posi=records_posi, record_strs=record_strs, cash_array=cash_arrays, prog_size=len(progs))
