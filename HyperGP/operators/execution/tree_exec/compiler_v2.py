
import numpy as np
from functools import reduce
from HyperGP.libs.representation.TGP import TGPIndv
import operator, psutil
import HyperGP, math, time, itertools
from HyperGP import Tensor
import gc

class ExecutableExpr:
    def __init__(self, exec_list, pset, states):
        self.codes, self.rets = [], []
        
        self.exec_unit_len = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        self.exec_list = exec_list
        self.pset = pset
        self.states = states

    def __call__(self, input, device="cuda"):

        """parameter initialization"""
        constants, cash_array, prog_size, records_posi, x_len = self.states["constants"], self.states["cash_array"], self.states['prog_size'], self.states['records_posi'], self.states['x_len']
        
        if not isinstance(cash_array, np.ndarray):
            assert input.shape[1] == cash_array.shape[1], "{0}, {1}; {2}, {3}".format(input.shape, cash_array.shape, len(input), len(cash_array))
        
        output = [[] for i in range(prog_size)]
        records = [[] for i in range(len(records_posi))]
        f_avec = [self.pset.genFunc(f_str) for f_str in self.pset.primitiveSet]
        
        def assign(x):
            return x

        # It will be called if a terminal node as a tree.    
        f_avec.append(assign)

        dtype_size = input.dtype.itemsize if not isinstance(input, Tensor) else HyperGP.sizeof(input.dtype)


        def prob(x):
            return reduce(operator.mul, x, 1)
        input_len = prob(input.shape)
        sizeof = prob(input.shape[1:]) * dtype_size
        data_size = int(sizeof / dtype_size)

        if device.startswith("cuda"):
            cur_free_m = HyperGP.src.ndarray.gpu().cuda_mem_available(0) - 128 - input.shape[1] * dtype_size * prog_size
            assert cur_free_m > 0, "no enough cuda memory,  {A} GiB is needed.".format(-(cur_free_m / (1024. ** 3)))
        else:
            cur_free_m = psutil.virtual_memory().free - 1 - sizeof * prog_size

        batch_num = 1
        if sizeof * x_len + input_len > (cur_free_m):
            batch_num = math.ceil((sizeof * x_len + input_len) / (cur_free_m))
        mid_output = {}
        output_segs = [[] for i in range(prog_size)]

        mid_output.update({-(i + 1): constants[i] for i in range(len(constants))})
        

        records = np.empty(shape=(len(records_posi), data_size))
        
        batch_init, batch_last = 0, input.shape[1]
        batch_size = int(input.shape[1] / batch_num)
        st = time.time()
        for z in range(batch_num - 1, -1, -1):
            mid_output = {-(i + 1): constants[i] for i in range(len(constants))}
            batch_range = slice(batch_init + batch_size * z, batch_last)
            # print("batch: ", z,  batch_num, HyperGP.src.ndarray.gpu().cuda_mem_available(0) / (1024. ** 3), batch_range.stop - batch_range.start)
            if not isinstance(cash_array, np.ndarray):
                for i in range(len(cash_array)):
                    mid_output[i + len(input) + prog_size] = cash_array[i][batch_range]
            for i in range(len(input)):
                mid_output[i] = Tensor(input[i][batch_range])
            
            idx = 0
            execs_size = len(self.exec_list)
            # print('2---------------------------------------------------', execs_size / self.exec_unit_len, batch_num, time.time() - st)
            # new_output = HyperGP.tensor.empty(shape=input.shape)

            while idx < execs_size:
                arity = self.exec_list[idx + 1]
                # print(self.exec_list[idx], arity, self.exec_list[idx + 2: idx+2+arity])
                # print('11111', f_avec[self.exec_list[idx]])
                # mid_output[0].device.cc()
                # mid_output[0].device.ewise_add(mid_output[0].cached_data._handle, mid_output[0].cached_data._handle, new_output.cached_data._handle, mid_output[0].cached_data._offset, mid_output[0].cached_data._offset)
                mid_output[self.exec_list[idx + arity + 2]] = f_avec[self.exec_list[idx]](*[mid_output[i] for i in self.exec_list[idx+2:idx+2+arity]])
                idx += self.exec_unit_len

            # print('1---------------------------------------------------', batch_num, time.time() - st)
            for i in range(prog_size):
                if len(mid_output[len(input) + i].shape) == 0 or mid_output[len(input) + i].shape[0] == 1:
                    output_segs[i].append(mid_output[len(input) + i].copy())
                output_segs[i].append(mid_output[len(input) + i].copy())

            # print("batch: ", z,  batch_num, HyperGP.src.ndarray.gpu().cuda_mem_available(0) / (1024. ** 3), batch_range.stop - batch_range.start)
            
                
            batch_last = batch_init + batch_size * z
        # print('0---------------------------------------------------', batch_num, time.time() - st)
        # output = [HyperGP.concatenate(tuple(output)) for out in output_segs]
        output = HyperGP.concatenate(tuple(itertools.chain.from_iterable(output_segs))).reshape((prog_size, -1))
        # print('---------------------------------------------------', time.time() - st)
        return output, records

    def __str__(self):
        return str(self.codes)

def compile_v2(exec_list, pset, states):
    return ExecutableExpr(exec_list, pset, states)