
import numpy as np
from functools import reduce
import operator, psutil
import HyperGP, math, time, itertools
from .... import Tensor

import os
if os.environ.get('READTHEDOCS') != 'True':
    from ....src import executor

class ExecutableExpr:
    def __init__(self, exec_list, pset, states):
        self.codes, self.rets = [], []
        
        self.exec_unit_len = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        self.exec_list = exec_list
        self.pset = pset
        self.states = states
        def assign(x):
            return x
        self.f_avec = [self.pset.genFunc(f_str) for f_str in self.pset.primitiveSet]
        self.f_avec.append(assign)
        self.func_list = [None] * (pset.func_count - len(f_avec)) + [f.func.exec_number if hasattr(f, "func") and hasattr(f.func, "exec_number") else -2 for f in self.f_avec] + [-1]


    def __call__(self, input, device="cuda"):

        """parameter initialization"""
        constants, cash_array, prog_size, records_posi, x_len = self.states["constants"], self.states["cash_array"], self.states['prog_size'], self.states['records_posi'], self.states['x_len']
        
        if not isinstance(cash_array, np.ndarray):
            assert input.shape[1] == cash_array.shape[1], "{0}, {1}; {2}, {3}".format(input.shape, cash_array.shape, len(input), len(cash_array))
        
        # It will be called if a terminal node as a tree.    
        dtype_size = input.dtype.itemsize if not isinstance(input, Tensor) else HyperGP.sizeof(input.dtype)


        def prob(x):
            return reduce(operator.mul, x, 1)
        input_len = prob(input.shape)
        sizeof = prob(input.shape[1:]) * dtype_size
        data_size = int(sizeof / dtype_size)

        if device.startswith("cuda"):
            cur_free_m = HyperGP.src.ndarray.gpu().cuda_mem_available(0) - 128 - input.shape[1] * dtype_size * prog_size
            assert cur_free_m > 0, "no enough cuda memory,  {A} GiB is needed.".format(A=-(cur_free_m / (1024. ** 3)))
        else:
            cur_free_m = psutil.virtual_memory().free - 1 - sizeof * prog_size

        batch_num = 1
        if sizeof * x_len + input_len > (cur_free_m):
            batch_num = math.ceil((sizeof * x_len + input_len) / (cur_free_m))
        mid_output = {}
        output_segs = [[] for i in range(prog_size)]
        record_segs = [[] for i in records_posi]
        mid_output = {-(i + 1): float(constants[i]) for i in range(len(constants))}
        
        execs_size = len(self.exec_list)
        
        batch_init, batch_last = 0, input.shape[1]
        batch_size = int(input.shape[1] / batch_num)
        st = time.time()
        params = [[] * int(execs_size / self.exec_unit_len)] # No need for now
        for z in range(batch_num - 1, -1, -1):
            batch_range = slice(batch_init + batch_size * z, batch_last)
            cur_batch_size = batch_range.stop - batch_range.start
            
            mem_space = HyperGP.tensor.empty(shape=(x_len, cur_batch_size), dtype=input.dtype)
            if not isinstance(cash_array, np.ndarray):
                for i in range(len(cash_array)):
                    # mid_output[i + len(input) + prog_size] = cash_array[i][batch_range]
                    mem_space[i + len(input) + prog_size, :] = cash_array[i][batch_range]
            
            for i in range(len(input)):
                mem_space[i] = Tensor(input[i][batch_range])
                # mid_output[i] = Tensor(input[i][batch_range])
            
            idx = 0
            # print('2---------------------------------------------------', execs_size / self.exec_unit_len, batch_num, time.time() - st)
            # new_output = HyperGP.tensor.empty(shape=input.shape)

            # HyperGP.MOD_SET("STATIC")
            # print("HyperGP.tensor.MOD: ", HyperGP.tensor.MOD)
            last_exec = 0
            while idx < execs_size:
                arity = self.exec_list[idx + 1]
                # print(self.exec_list[idx], arity, self.exec_list[idx + 2: idx+2+arity])
                # mid_output[0].device.cc()
                # mid_output[0].device.ewise_add(mid_output[0].cached_data._handle, mid_output[0].cached_data._handle, new_output.cached_data._handle, mid_output[0].cached_data._offset, mid_output[0].cached_data._offset)
                if self.func_list[self.exec_list[idx]] == -2 or len(self.f_avec[self.exec_list[idx]].kwargs) > 0:# [ ] TODO: can not process the function with default parameters
                    if idx - last_exec > 0:
                        mem_space.wait()
                        executor.exec_partial(
                            mem_space.realize_cached_data._handle,
                            self.exec_list[last_exec:idx], 
                            constants, params, self.func_list, 
                            self.exec_unit_len, cur_batch_size)
                    mem_space[int(self.exec_list[idx + arity + 2])] = self.f_avec[self.exec_list[idx]](*[mem_space[int(i)] if i >= 0 else mid_output[int(i)] for i in self.exec_list[idx+2:idx+2+arity]])
                    
                    last_exec = idx + self.exec_unit_len
                # mid_output[self.exec_list[idx + arity + 2]] = self.f_avec[self.exec_list[idx]](*[mid_output[i] for i in self.exec_list[idx+2:idx+2+arity]])
                
                idx += self.exec_unit_len
            if last_exec < execs_size:
                executor.exec_partial(
                    mem_space.realize_cached_data._handle,
                    self.exec_list[last_exec:], 
                    constants, params, self.func_list, 
                    self.exec_unit_len, cur_batch_size)
            # print('1---------------------------------------------------', batch_num, time.time() - st)
            
            if batch_num > 1:
                for i in range(prog_size):
                    output_segs[i].append(mem_space[len(input) + i])
                    record_segs[i].append(mem_space[list(records_posi)])
            else:
                output = mem_space[len(input):len(input) + prog_size]
                records = mem_space[list(records_posi)]

            batch_last = batch_init + batch_size * z
        if batch_num > 1:
            output = HyperGP.concatenate(tuple(itertools.chain.from_iterable(output_segs))).reshape((prog_size, -1))
            records = HyperGP.concatenate(tuple(itertools.chain.from_iterable(record_segs))).reshape((len(records_posi), -1))

        return output, records

    def __str__(self):
        return str(self.codes)

def compile_v2(exec_list, pset, states):
    return ExecutableExpr(exec_list, pset, states)
