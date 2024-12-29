import time
from HyperGP.libs.representation.TGP import TGPIndv

from HyperGP.libs.primitive_set import PrimitiveSet
from HyperGP.mods.tree2graph import ExecutableGen
import itertools
import numpy as np
from functools import reduce
import operator
import math
from HyperGP.mods.cash_manager import CashManager
from HyperGP.base.base_struct import States
import psutil
from HyperGP import TensorOps
import HyperGP

class ExecutableExpr:
    def __init__(self, exec_list, pset, states):
        self.codes, self.rets = [], []
        
        self.exec_unit_len = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        self.exec_list = exec_list
        self.pset = pset
        self.states = states

        # for execs_ in exec_list:
        #     for idx in range(0, len(execs_), 1000):
        #         execs = execs_[idx: idx+1000 if idx+1000 < len(execs_) else len(execs_)]
                
        #         code = '[' + ", ".join((str(pset.genFunc(pset.primitiveSet[exec_[0]])) + '(' + ",".join(
        #             'x[%d]' % (exec_[2 + j]) for j in range(exec_[1])) + ')') if exec_[0] >= 0 else 'x[%d]'%(exec_[2]) for i, exec_ in enumerate(execs)) + ']'
        #         ret = [exec_[2 + exec_[1]] for i, exec_ in enumerate(execs)]
        #         code = "lambda {args}: {code}".format(args='x', code=code)
        #         self.rets.append(ret)
        #         self.codes.append(code)

    def __call__(self, input):
        
        cur_free_m = psutil.virtual_memory().free

        """parameter initialization"""
        constants, cash_array, prog_size, records_posi, x_len = self.states["constants"], self.states["cash_array"], self.states['prog_size'], self.states['records_posi'], self.states['x_len']
        
        if not isinstance(cash_array, np.ndarray):
            assert input.shape[1] == cash_array.shape[1], "{0}, {1}; {2}, {3}".format(input.shape, cash_array.shape, len(input), len(cash_array))
            
        output = [[] for i in range(prog_size)]
        records = [[] for i in range(len(records_posi))]
        f_avec = [self.pset.genFunc(f_str) for f_str in self.pset.primitiveSet]
            
        def prob(x):
            return reduce(operator.mul, x, 1)

        sizeof = prob(input.shape[1:]) * HyperGP.sizeof(input.dtype)
        data_size = int(sizeof / HyperGP.sizeof(input.dtype))
        batch_num = 1
        if sizeof * x_len + prob(input.shape) > (cur_free_m / 2):
            batch_num = math.ceil((sizeof * x_len + prob(input.shape)) / (cur_free_m / 2))
        mid_output = {}
        output_segs = [[] for i in range(prog_size)]
        # print('batch_num: ', batch_num)

        mid_output.update({-(i + 1): constants[i] for i in range(len(constants))})
        # funcs = []
        # for i in range(0, len(self.exec_list), self.exec_unit_len):
        #     funcs.append(self.pset.genFunc(self.pset.primitiveSet[self.exec_list[i]]))
        # print("------!1!-------")
        st = time.time()
        
        # print("------!2!-------")

        records = np.empty(shape=(len(records_posi), data_size))
        # print("------!3!-------")
        # print('constants:', constants)
        # print(prog_size * data_size / 1024 ** 3, cur_free_m / 1024 ** 3)
        
        # for i in range(0, len(self.exec_list), self.exec_unit_len):
        #     print('exec: ', self.exec_list[i: i + self.exec_unit_len])
        batch_init, batch_last = 0, input.shape[1]
        batch_size = int(input.shape[1] / batch_num)
        st = time.time()
        for z in range(batch_num - 1, -1, -1):
            # print('z: ', z, batch_num)
            # print("----------------------4----------------------")
            batch_range = slice(batch_init + batch_size * z, batch_last)
            # print(batch_range)
            # print(z, batch_num, batch_size, input.shape[1], batch_num)
            if not isinstance(cash_array, np.ndarray):
                for i in range(len(cash_array)):
                    mid_output[i + len(input) + prog_size] = cash_array[i][batch_range]
            for i in range(len(input)):
                mid_output[i] = TensorOps(input[i][batch_range])
            
            # print("----------------------1----------------------")
            idx = 0
            execs_size = len(self.exec_list)
            while idx < execs_size:
                arity = self.exec_list[idx + 1]
                # print(self.exec_list[idx], arity, self.exec_list[idx + 2: idx+2+arity])
                # print('11111', f_avec[self.exec_list[idx]])
                mid_output[self.exec_list[idx + arity + 2]] = f_avec[self.exec_list[idx]](*[mid_output[i] for i in self.exec_list[idx+2:idx+2+arity]])
                idx += self.exec_unit_len
            # print("----------------------2----------------------")

            for i in range(prog_size):
                output_segs[i].append(mid_output[len(input) + i])
            # for i, posi in enumerate(records_posi):
            #     records[i][batch_range] = mid_output[posi][batch_range]
            # print("----------------------3----------------------")
                
            batch_last = batch_init + batch_size * z
        output = HyperGP.concatenate(tuple(itertools.chain.from_iterable(output_segs))).reshape((prog_size, -1))
        print('time: ', time.time() - st)
        return output, records

    def __str__(self):
        return str(self.codes)

def compile(exec_list, pset, states):
    return ExecutableExpr(exec_list, pset, states)

class ExecMethod:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No find '__call__' implement")

class ExecCPU(ExecMethod):
    def __init__(self):
        pass

    def __call__(self, progs, input:np.array, pset: PrimitiveSet, cashset:CashManager=None):

        st = time.time()
        num = 0
        # for prog in progs:
        #     assert len(prog.list()) != 1, "{A}, {B}".format(A=str(prog), B=len(prog.list()))
        #     print('prog:', prog)
        exec_set, states = ExecutableGen()(progs, pset, cashset)
        # exec_list = list(itertools.chain.from_iterable(exec_set))
        # num += len(exec_list)
        print('prog funcs after optimize: ', num, time.time() - st)


        st = time.time()
        expr = compile(exec_set, pset, states)
        print('t0: ', time.time() - st)
        st = time.time()
        output, records = expr(input)
        print('t1: ', time.time() - st)
            
        # print('t-1: ', time.time() - st)
        # assert 0==1
        # print(exprs)
        # st = time.time()
        # func_list = [pset.genFunc(pset.primitiveSet[exec_[0]]).func for i, exec_ in enumerate(exec_list)]
        #
        # print('t0: ', time.time() - st)
        # for i in range(len(exec_set)):
        #     print('len(exec_set): ', i, len(exec_set[i]))
        # st = time.time()
        # for i, exec_ in enumerate(exec_list):
        #     middle_output[exec_[2 + exec_[1]]] = func_list[i](*[middle_output[exec_[2 + j]] for j in range(exec_[1])])
        # print('t1: ', time.time() - st)
        return output, States(records_array=records, records_posi=states['records_posi'], record_strs=states['record_strs'])#[middle_output[i] for i in range(len(input), len(input) + len(progs), 1)], {record[1]:middle_output[record[0]] for record in record_set.values()}

