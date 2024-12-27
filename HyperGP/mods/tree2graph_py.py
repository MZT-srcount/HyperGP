from HyperGP.library.representation.TGP import TGPIndv
from HyperGP.library.primitive_set import PrimitiveSet
from HyperGP.base.func_basic import Constant
from HyperGP.base.base_struct import States
from .cash_manager import CashManager
import numpy as np
from ..src import pygp_utils
import itertools


"""[ ] TODO: Execution respond to get the output data of expression, there should be the ability to return the data need to record"""
class ExecutableGen:
    def __call__(self, progs:[TGPIndv], pset: PrimitiveSet, cash_manager: CashManager = None, ind_num=100):
        input_len_max = max([value.arity for key, value in pset.used_primitive_set.items()])
        id_init, id_p, id_max = len(pset.arguments) + len(progs), 0, 0
        init_output, output, constant_set, record_set = {}, {}, [], {}
        sym_set, exp_set, exp_iposi_set = {}, [], []
        cash_list, record_dict = [], {}

        nodes_list, idxs_list, init_posi = [[] for i in range(len(progs))], [[] for i in range(len(progs))], [0 for i in range(len(progs) + 1)]
    
        '''cash scan first, to find the subtree without cash'''
        if cash_manager is not None:
            for i, prog in enumerate(progs):
                # print(len(nodes_list), len(idxs_list), i)
                nodes_list[i], cash_set, idxs_list[i], symset_after_cash = cash_manager.getCash(prog)
                init_posi[i + 1] = len(nodes_list[i]) + init_posi[i]
                sym_set.update(symset_after_cash)
                for key, value in cash_set.items():
                    init_output[key] = (id_init, 0)
                    cash_list.append(value)
                    id_init += 1

        id_p = id_init
        
        import time
        
        st = time.time()
        # nodes_list_ = list(itertools.chain.from_iterable(nodes_list))
        # idxs_list_ = list(itertools.chain.from_iterable(idxs_list))
        nodes_arity, nodes_str, idxs = [], [], []
        cash_records = [[] for i in range(len(progs))]
        cash_num = 0
        for k, prog in enumerate(progs):
            prog_list = prog.list()
            nodes_arity.append([node.arity for node in prog_list])
            idxs.append([node.nodeval.idx for node in prog_list])
            nodes_str.append([str(node.nodeval) for node in prog_list])
            if cash_manager is not None:
                for i, node in enumerate(prog_list):
                     if ('cash_record' in node.states and node.states['cash_record']) or ('record' in node.states and node.states['record']):
                         cash_records[k].append(i)
                         node.states['cash_record'] = False
                         cash_num += 1
        id_init += cash_num
                # cash_records.extend(i for i, node in enumerate(prog_list) if 'cash_record' in node.states and node.states['cash_record'])
                # cash_records.extend(list(filter(lambda node: 'cash_record' in prog_list[i].states and prog_list[i].states['cash_record'], range(len(prog_list)))))
        # nodes_arity, nodes_str, idxs = [], [], []
        # for i, node in enumerate(nodes_list_):
        #     idxs.append(node.nodeval.idx)
        #     nodes_arity.append(node.arity)
        #     nodes_str.append(str(node.nodeval))
            
        #     if cash_manager is not None and 'cash_record' in node.states and node.states['cash_record']:
        #         node.states['cash_record'] = False
        #         # assert sym in output
        #         cash_records[i] = True
        res = pygp_utils.tree2graph(nodes_arity, nodes_str, 
                                          idxs_list, idxs, [len(pset.arguments), id_init, input_len_max])
        print('pre_time: ', time.time() - st)
        assert 0==1
        for i, prog in enumerate(progs):
            if i % ind_num == 0:
                output.update(init_output)

            node_list = prog.list()
            idx_list = list(range(len(node_list)))
            c_list = prog.list(childs=True)[0]

            root = node_list[0]

            '''cash scan first, to find the subtree without cash'''
            if cash_manager is not None:
                node_list, idx_list = nodes_list[i], idxs_list[i]#cash_manager.getCash(prog)


            '''compute the subtrees'''
            for node_idx in range(len(node_list) - 1, -1, -1):
                node = node_list[node_idx]
                origin_idx = idx_list[node_idx]
                if node.arity != 0:
                    '''a func node'''
                    childs = [prog[c_idx] for c_idx in c_list[origin_idx]]
                    child_vals, child_ly = [], []
                    sym = str(node.nodeval) + '('
                    for child in childs:
                        sym += sym_set[child] + ', '
                        child_vals.append(output[sym_set[child]][0])
                        child_ly.append(output[sym_set[child]][1])
                    sym = sym[:-2] + ')'
                    sym_set[node] = sym

                    if sym in output and node != root:
                        '''if there is an existed same sub-expression'''
                        pass
                    else:
                        '''else, generate a new expunit'''
                        expunit = []
                        expunit.extend([pset.used_primitive_set[node.nodeval.name].idx, len(childs)])
                        expunit.extend(child_vals)
                        if node == root:
                            output_posi = len(pset.arguments) + i
                        else:
                            while id_p in record_dict:
                                id_p += 1
                            output_posi = id_p
                            id_p += 1
                        expunit.append(output_posi)
                        cur_layer = max(child_ly) + 1
                        output[sym] = (output_posi, cur_layer)

                        '''align'''
                        expunit.extend([0] * (input_len_max + 3 - len(expunit)))
                        assert len(expunit) == input_len_max + 3

                        '''set the expunit to the corresponding layer'''
                        if cur_layer - 1 >= len(exp_set):
                            # exp_iposi_set.append([0])
                            exp_set.append([expunit])
                        else:
                            # exp_iposi_set[cur_layer - 1].append(exp_iposi_set[cur_layer - 1][-1] + len(exp_set[cur_layer - 1][-1]))
                            exp_set[cur_layer - 1].append(expunit)

                    if cash_manager is not None and 'cash_record' in node.states and node.states['cash_record']:
                        node.states['cash_record'] = False
                        # assert sym in output
                        record_set[sym] = (output[sym][0], sym)
                        record_dict[output[sym][0]] = 1

                else:
                    '''a terminal node'''
                    if isinstance(node.nodeval, Constant):
                        if node.nodeval not in output:
                            cval = node.nodeval.val
                            constant_set.append(cval)
                            output[str(node.nodeval)] = (-len(constant_set), 0)
                        sym_set[node] = str(node.nodeval)
                    elif hasattr(node.nodeval, 'idx'):
                        if node.nodeval not in output:
                            idx = pset.used_terminal_set[str(node.nodeval)].idx
                            output[str(node.nodeval)] = (idx, 0)
                        sym_set[node] = str(node.nodeval)
                    else:
                        cval = pset.used_terminal_set[str(node.nodeval)].func()
                        if str(cval) not in output:
                            constant_set.append(cval)
                            output[str(cval)] = (-len(constant_set), 0)
                        sym_set[node] = str(cval)
            '''no any node need to compute, means that the root node is cashed or there is no any node in this prog'''
            if len(node_list) == 0:
                if 0 == len(exp_set):
                    exp_set.append([[-1, 1, output[sym_set[prog.list()[0]]][0], len(pset.arguments) + i]])
                else:
                    exp_set[0].append([-1, 1, output[sym_set[prog.list()[0]]][0], len(pset.arguments) + i])
            '''only one node need to compute, means that the remain node is a leaf node'''
            if len(node_list) == 1:
                if 0 == len(exp_set):
                    exp_set.append([-1, 1, output[sym_set[node_list[0]][0]], len(pset.arguments) + i])
                else:
                    exp_set[0].append([-1, 1, output[sym_set[node_list[0]][0]], len(pset.arguments) + i])

        return exp_set, States(constants=constant_set, x_len=id_max, record_set=record_set, cash_array=np.array(cash_list), prog_size=len(progs))
