
import operator
import time
from .... import Tensor, States
from ....src import executor, NDArray
import numpy as np

class ExecutableExpr:
    def __init__(self, exec_list, pset, states, func_list = None):
        self.prog_size, self.constants, self.x_len, self.records_posi, self.cash_array = states['prog_size'], states["constants"], states["x_len"], states["records_posi"], states["cash_array"]
        self.execs_layer_info = states["layer_info"]
        self.exec_unit_len = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        self.pset = pset
        self.states = states
        self.exec_list = exec_list
        if func_list is None:
            f_avec = [self.pset.genFunc(f_str) for f_str in self.pset.primitiveSet]
            self.func_list = [None] * (pset.func_count - len(f_avec)) + [f.func.exec_number if hasattr(f, "func") and hasattr(f.func, "exec_number") else -2 for f in f_avec] + [-1]
        else:
            self.func_list = func_list
        
    def __call__(self, input):
        '''hyper-parameters of GPU run'''
        outputs = NDArray.make(shape=(self.prog_size, input.shape[1]), dtype=input.dtype)#gpu().Array(len(progs) * input.shape[1])
        records = NDArray.make(shape=(len(self.records_posi), input.shape[1]) if len(self.records_posi) > 0 else (1,), dtype=input.dtype)#gpu().Array(len(records_posi) * input.shape[1] if len(records_posi) > 0 else 1)
        paras = (self.exec_unit_len, self.x_len, len(self.pset.arguments) + self.prog_size, self.prog_size, len(self.pset.arguments))
        
        if isinstance(input, Tensor):
            new_input = input
        else:
            new_input = Tensor(input)
            # if isinstance(input, list):
            #     new_input = np.array(input)
            # else:
            #     new_input = input
            # executor.exec_cpuinput(self.exec_list, self.execs_layer_info, 
            #             self.constants, new_input, self.cash_array, self.func_list, 
            #             self.records_posi, outputs._handle, records._handle, paras)
            
        new_input = executor.InputCuda(input.realize_cached_data.ptr(), new_input.realize_cached_data.offset, input.shape)
        executor.exec_gpuinput(self.exec_list, self.execs_layer_info, 
                    self.constants, new_input, 
                    self.cash_array, self.func_list,
                    self.records_posi, outputs._handle, records._handle, paras)
        
        outputs=Tensor(NDArray.make(handle=outputs._handle, shape=tuple([self.prog_size] + list(input.shape[1:])), dtype=outputs.dtype))
        records=Tensor(NDArray.make(handle=records._handle, shape=tuple([len(self.records_posi)] + list(input.shape[1:])), dtype=records.dtype))
        
        # print('et: ', time.time() - st)
        return outputs, States(records_array=records, records_posi=self.records_posi, records_str=self.states['record_strs'])
    def __str__(self):
        return str(self.codes)

def compile_v1(exec_list, pset, states, func_list=None):
    return ExecutableExpr(exec_list, pset, states, func_list)