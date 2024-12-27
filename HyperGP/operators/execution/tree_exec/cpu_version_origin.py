import time
from HyperGP.library.representation.TGP import TGPIndv

from HyperGP.library.primitive_set import PrimitiveSet
from HyperGP.mods.tree2graph import ExecutableGen
import itertools
import numpy as np
from functools import reduce
import operator
import math
from HyperGP.mods.cash_manager import CashManager
from HyperGP.base.base_struct import States

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

        constants, cash_array, prog_size, records_posi, x_len = self.states["constants"], self.states["cash_array"], self.states['prog_size'], self.states['records_posi'], self.states['x_len']
        output = [[] for i in range(prog_size)]
        records = [[] for i in range(len(records_posi))]
        
        def prob(x):
            return reduce(operator.mul, x, 1)
        sizeof = prob(input.shape[1:]) * 8
        data_size = int(sizeof / 8)
        batch_num = 1
        if sizeof * x_len > 2 * 1024 ** 3:
            batch_num = math.ceil((sizeof * x_len) / (2 * 1024 ** 3))
        mid_output = {}
        print('batch_num: ', batch_num)
        mid_output.update({-(i + 1): constants[i] for i in range(len(constants))})
        
        funcs = []
        for i in range(0, len(self.exec_list), self.exec_unit_len):
            funcs.append(self.pset.genFunc(self.pset.primitiveSet[self.exec_list[i]]))
        st = time.time()
        input = np.array_split(input, batch_num, 1)
                
        if cash_array.shape[0] > 0:
            cash_array = np.array_split(cash_array, batch_num, 1)
        
        cur_size = 0
        output = np.empty(prog_size * data_size)
        records = np.empty(len(records_posi) * data_size)
        
        idx = 0
        execs, exec_input, exec_output = [], [], []
        while idx < len(self.exec_list):
            input_len = self.exec_list[idx + 1]
            execs.append(self.exec_list[idx])
            exec_input.append(self.exec_list[idx+2:idx+2 + input_len])
            exec_output.append(self.exec_list[idx + 2 + input_len])
            idx += self.exec_unit_len
        for z in range(batch_num - 1, -1, -1):
            batch_size = input[z].shape[1]
            print(z, batch_num)
            if not isinstance(cash_array, np.ndarray):
                assert input[z].shape[1] == cash_array[z].shape[1], "{0}, {1}; {2}, {3}".format(input[z].shape, cash_array[z].shape, len(input), len(cash_array))
                for i in range(len(cash_array[z])):
                    mid_output[i + len(input[z]) + prog_size] = cash_array[z][i]
            for i in range(len(input[z])):
                mid_output[i] = input[z][i]
            
            print("----------------------1----------------------")
            idx = 0
            execs_size = len(self.exec_list) / self.exec_unit_len
            while idx < execs_size:
                mid_output[exec_output[idx]] = funcs[execs[idx]](*[mid_output[i] for i in exec_input[idx]])
                # input_len = self.exec_list[i + 1]
                # input_list = self.exec_list[i + 2: i + 2 + input_len]
                # # print(i, self.exec_list[i:i+5])
                # mid_output[self.exec_list[i + 2 + input_len]] = funcs[idx](*[mid_output[i] for i in input_list])
                idx += 1
            print("----------------------2----------------------")
            # for i, code in enumerate(self.codes):
            #     mid_output[z].update({self.rets[i][k]: arr for k, arr in enumerate(eval(code, self.pset.context, {})(mid_output[z]))})
            # init_batch_posi = z * batch_size
            # for i in range(prog_size):
            #     init_posi = i * data_size + init_batch_posi
            #     output[init_posi: init_posi + batch_size] = mid_output[len(input[z]) + i]
            #     # output[i].extend(mid_output[len(input[z]) + i])
            # for i in range(len(records_posi)):
            #     init_record_posi = i * data_size
            #     output[init_record_posi + init_batch_posi: init_record_posi + init_batch_posi + batch_size] = mid_output[records_posi[i]]
                # records[i].extend(mid_output[records_posi[i]])
            print("----------------------3----------------------")
            cur_size += batch_size
        
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

    def __call__(self, progs, input:np.array, pset: PrimitiveSet, cashset:CashManager=None, precision=8):

        if input.dtype == np.float32 and precision==8:
            precision = 4
            raise UserWarning('precision is set to 8 while the input is in np.float32,'
                              ' the precision has been automatically changed to float(4)')
        elif input.dtype == np.float64 and precision==4:
            precision = 8
            raise UserWarning('precision is set to 4 while the input is in np.float64,'
                              ' the precision has been automatically changed to double(8)')



        st = time.time()
        num = 0
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

