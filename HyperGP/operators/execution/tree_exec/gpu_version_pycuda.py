import time

from HyperGP.libs.primitive_set import PrimitiveSet
import pycuda.driver as cuda
from HyperGP.src.cuda_backend import memcopy_2D, mod
from pycuda.autoinit import context as sync
from HyperGP.mods.tree2graph import ExecutableGen
import itertools
import numpy as np
import random
from HyperGP.mods.cash_manager import CashManager
from .cpu_version import ExecMethod

class ExecGPU(ExecMethod):
    def __init__(self):
        pass
    """
    input: (n_terms, dataset_size)
    """
    def __call__(self, progs, input:np.array, pset: PrimitiveSet, cashset:CashManager=None, precision=8):
        if input.dtype == np.float32 and precision==8:
            precision = 4
            raise UserWarning('precision is set to 8 while the input is in np.float32,'
                              ' the precision has been automatically changed to float(4)')
        elif input.dtype == np.float64 and precision==4:
            precision = 8
            raise UserWarning('precision is set to 4 while the input is in np.float64,'
                              ' the precision has been automatically changed to double(8)')

        ''''''
        st = time.time()
        exec_set, states = ExecutableGen()(progs, pset, cashset)

        constants, x_len, records_posi, cash_array = states["constants"], states["x_len"], states["records_posi"], states["cash_array"]

        exec_list = list(itertools.chain.from_iterable(exec_set))
        exec_list = [val for exec in exec_list for val in exec]
        exec_unit_len = max([value.arity for key, value in pset.used_primitive_set.items()]) + 3
        # exec_iposi_list = list(itertools.chain.from_iterable(exec_iposi_set))
        print(type(exec_set), np.shape(exec_list), time.time() - st)

        st = time.time()
        '''hyper-parameters of GPU run'''
        exec_len = 0#len(exec_iposi_list)
        for execs in exec_set:
            exec_len += len(execs)
        max_threads_per_block = cuda.Device(0).max_threads_per_block#cuda.device_attribute.MAX_THREADS_PER_BLOCK#cuda.Device(0).make_context().get_device().MAX_THREADS_PER_BLOCK
        # thread_num  = 512
        warp_size   = 32
        # min_batch   = 0
        # max_batch   = 0
        block_num   = 50

        feature_num, d_size = np.shape(input)
        record_list = records_posi
        record_size = len(record_list)

        # output_there = {}
        # for i in range(len(exec_list)):
        #     if exec_list[i] == 102:
        #         print('exec_list[i]', i, exec_list[i], exec_list[i-4:i+1])
        #     if exec_list[i] >= feature_num and exec_list[i] < feature_num + len(progs):
        #         # print('exec_list[i]', i, exec_list[i], exec_list[i-4:i+1])
        #         output_there[exec_list[i]] = None
        # assert len(output_there) == len(progs), "%d, %d"%(len(output_there), len(progs))


        '''
        data parallel happened between blocks
        exec parallel happened between warps
        
        warp_num is the min parallable exec_unit num
        '''
        if d_size < warp_size * block_num:
            '''no suitable for GPU computing, use CPU method instead'''
            raise NotImplementedError('A CPU method is called here, not implement yet')
        else:
            print('min_exec_len: ', [len(execs) for execs in exec_set])
            min_exec_len = min([len(execs) for execs in exec_set])
            warp_num = min_exec_len if min_exec_len * warp_size < max_threads_per_block else max_threads_per_block / warp_size
            thread_num = warp_num * warp_size

        batch = int(input.shape[1] / (block_num * 2 * warp_size)) if input.shape[1] > (block_num * 2 * warp_size) else 1

        d_batch = int(np.ceil(d_size / batch))
        # print('input: ', [input[0][d_size - d_batch * i if d_batch * i > d_size else d_batch * i] for i in range(batch)])

        # d_perblock = d_batch / block_num + d_batch % block_num

        assert feature_num == len(pset.arguments)

        output = np.empty((len(progs), d_size)).astype(np.float32 if precision == 4 else np.float64)
        record_output = np.empty((record_size, d_size)).astype(np.float32 if precision == 4 else np.float64)


        '''cuda memory alloc'''
        exec_gpu = cuda.mem_alloc(len(exec_list) * precision)
        # exec_iposi_gpu = cuda.mem_alloc(len(exec_iposi_list) * precision)
        const_gpu = cuda.mem_alloc((len(constants) if len(constants) > 0 else 1) * precision)
        (input_gpu, input_pitch) = cuda.mem_alloc_pitch(int(d_batch * precision), x_len * 2, precision)

        '''cuda memory transfer'''
        cuda.memcpy_htod(exec_gpu, np.array(exec_list))
        # cuda.memcpy_htod(exec_iposi_gpu, np.array(exec_iposi_list))
        cuda.memcpy_htod(const_gpu, np.array(constants))
        memcopy_2D(input_gpu, input_pitch,
                   input, d_size * precision,
                   int(d_batch * precision), feature_num
                   )#存疑，需测试:d_size or d_batch?
        cash_len = np.shape(cash_array)[0]
        if(cash_len > 0):
            memcopy_2D(input_gpu, input_pitch,
                       cash_array, cash_len * precision,
                       int(d_batch * precision), np.shape(cash_array)[1],
                       dst_y_offset=feature_num + len(progs),
                       )
            assert 0==1

        '''cuda preparation'''
        streams = [cuda.Stream() for i in range(2)]
        execution_GPU = mod.get_function('execution_GPU')

        print('t1: ', time.time() - st, batch)
        st = time.time()
        t = 0
        '''iteration'''
        d_offset = d_batch
        for i in range(batch):
            if d_offset * i + d_batch > d_size:
                d_batch = d_size - d_batch * i
            '''execution'''
            if i < batch - 1:
                d_batch_next = d_batch if d_offset * (i + 1) + d_batch <= d_size else d_size - d_batch * (i + 1)
                memcopy_2D(input_gpu, input_pitch,
                           input, d_size * precision,
                           d_batch_next * precision, feature_num,
                           dst_y_offset=x_len * ((i + 1) % 2),
                           src_x_offset=(i + 1) * d_offset * precision, stream=streams[(i + 1) % 2]
                           )#存疑，需测试:d_size or d_batch?

                if cash_len > 0:
                    memcopy_2D(input_gpu, input_pitch,
                               cash_array, cash_len * precision,
                               d_batch_next * precision, np.shape(cash_array)[1],
                               dst_y_offset=x_len * ((i + 1) % 2) + feature_num + len(progs),
                               src_x_offset=(i + 1) * d_offset * precision, stream=streams[(i + 1) % 2]
                               )
            # st_1 = time.time()
            # sync.synchronize()

            execution_GPU(exec_gpu, np.int32(exec_unit_len), np.int32(exec_len), input_gpu,
                          cuda.In(np.array((input_pitch, x_len * (i % 2)), dtype=np.int64)), np.int32(d_batch),
                          const_gpu, block=(thread_num, 1, 1), grid=(block_num, 1, 1), stream=streams[i % 2])
            #
            # sync.synchronize()
            # t += time.time() - st_1
            memcopy_2D(output, d_size * precision,
                       input_gpu, input_pitch,
                       d_batch * precision, len(progs),
                       dst_x_offset=i * d_offset * precision,
                       src_y_offset=(i % 2) * x_len + feature_num, stream=streams[i % 2]
                       )  # 存疑，需测试:d_size or d_batch?

            for j, item in enumerate(record_list):
                memcopy_2D(record_output, d_size * precision,
                           input_gpu, input_pitch,
                           d_batch * precision, 1,
                           dst_x_offset=i * d_offset * precision,
                           dst_y_offset=j,
                           src_y_offset=(i % 2) * x_len + item[0], stream=streams[i % 2]
                           )  # 存疑，需测试:d_size or d_batch?

        sync.synchronize()
        # print(output)
        # print('t: ', t)

        for i, item in enumerate(record_list):
            record_set[item[1]] = record_output[i]

        print('t2: ', time.time() - st)
        return output, record_set


"""TEST"""
if "__name__" == "__main__":
    pass


