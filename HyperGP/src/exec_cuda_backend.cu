#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include<Array.h>
#include <algorithm>

#include <iostream>
#include <cmath>
#include <sys/mman.h>

namespace py=pybind11;

struct InputCuda{
    size_t ptr;
    std::vector<size_t> shape;
    int offset;
    InputCuda(size_t input_ptr, int offset, std::vector<size_t>& input_shape){
        this->ptr = input_ptr;
        this->offset = offset;
        shape = input_shape;
    }
};

namespace pygp_exec{

#define MEM_CAPACITY (10.0 * 1024 * 1024 * 1024)

template<typename scalar_t>
__device__ void add_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        scalar_t output_constant = constant[0] + constant[1];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[0] + input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[1] + input[0][i];
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = input[0][i] + input[1][i];
        }
        //if(blockIdx.x == 0 && threadIdx.x == 0){
        //    printf("d_perblock: %d, %f, %f, %f\\n", d_perblock, output[1], input[0][1], input[1][1]);
        //}
    }
    
}

template<typename scalar_t>
__device__ void sub_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        scalar_t output_constant = constant[0] - constant[1];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[0] - input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[1] - input[0][i];
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = input[0][i] - input[1][i];
        }
    }
}

template<typename scalar_t>
__device__ void mul_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        scalar_t output_constant = constant[0] * constant[1];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[0] * input[1][i];
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = constant[1] * input[0][i];
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = input[0][i] * input[1][i];
        }
    }
}

template<typename scalar_t>
__device__ void div_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0 && exec_unit[3] < 0){
        scalar_t output_constant = constant[0] / constant[1];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else if (exec_unit[2] < 0 && exec_unit[3] >= 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            if(input[1][i] == 0){
                output[i] = constant[0];
            }
            else{
                output[i] = constant[0] / input[1][i];
            }
        }
    }
    else if (exec_unit[2] >= 0 && exec_unit[3] < 0){
        for(int i = init_id; i < d_perblock; i += thread_num){
            if(input[0][i] == 0){
                output[i] = constant[1];
            }
            else{
                output[i] = constant[1] / input[0][i];
            }
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            if(input[1][i] == 0){
                output[i] = input[0][i];
            }
            else{
                output[i] = input[0][i] / input[1][i];
            }
        }
    }
}

template<typename scalar_t>
__device__ void assign_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0){
        scalar_t output_constant = constant[0];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = input[0][i];
        }
    }
}

template<typename scalar_t>
__device__ void copy_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[3] < 0){
        scalar_t output_constant = constant[3];
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
        for(int i = init_id; i < d_perblock; i += thread_num){
            input[0][i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = input[1][i];
        }
        for(int i = init_id; i < d_perblock; i += thread_num){
            input[0][i] = input[1][i];
        }
    }
}

template<typename scalar_t>
__device__ void sin_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0){
        scalar_t output_constant = sin(double(constant[0]));
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = sin(double(input[0][i]));
        }
    }
}

template<typename scalar_t>
__device__ void cos_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0){
        scalar_t output_constant = cos(double(constant[0]));
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = cos(double(input[0][i]));
        }
    }
}

template<typename scalar_t>
__device__ void log_fabs(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0){
        scalar_t output_constant = log(fabs(double(constant[0])));
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = log(fabs(double(input[0][i])));
        }
    }
}

template<typename scalar_t>
__device__ void exp_d(int* exec_unit, scalar_t** input, scalar_t* output, scalar_t* constant, int d_perblock, int thread_num, int init_id){
    if (exec_unit[2] < 0){
        scalar_t output_constant = exp(double(constant[0]));
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = output_constant;
        }
    }
    else {
        for(int i = init_id; i < d_perblock; i += thread_num){
            output[i] = exp(double(input[0][i]));
        }
    }
}

template<typename scalar_t>
__global__ void test(scalar_t* d_set_, size_t d_pitch, int d_offset){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        printf("%d, %l\n", d_offset, d_pitch);
        scalar_t* d_set = (scalar_t*)(d_set_ + d_pitch * d_offset / sizeof(scalar_t));
        for(int i = 0; i < 10; ++i){
            printf("d_set: %f\n", d_set[i]);
        }
    }
}

template<typename scalar_t>
__global__ void execution_GPU(int* exec_units, size_t exec_unit_len, size_t* layers, size_t layer_num, scalar_t* dataset, size_t d_pitch, size_t d_offset_whole, size_t dlen, scalar_t* constants){
    // printf("d_offset_whole: %f\n", d_offset_whole);
    scalar_t* d_set = (scalar_t*)(dataset + d_offset_whole * d_pitch / sizeof(scalar_t));
    size_t warp_size = 32, wid = threadIdx.x / warp_size;
    size_t block_num = gridDim.x, warp_num = blockDim.x / warp_size, eid, execnum_periter, wnum_exec, tid;
    size_t d_perblock = dlen / block_num;
    size_t d_offset = d_perblock * blockIdx.x;
    if(blockIdx.x < dlen % block_num){
        d_perblock += 1;
        d_offset += blockIdx.x;
    }
    else{
        d_offset += dlen % block_num;
    }
    size_t data_init = 0;
    size_t exec_id;
    
    int* exec_unit;
    int init_exec = 0;
    for(int z = 0; z < layer_num; ++z){
        size_t remain_execs = layers[z], exec_len = layers[z];
        while(remain_execs > 0){
            
            execnum_periter = warp_num;
            if (execnum_periter > remain_execs){
                execnum_periter = remain_execs;
            }
            eid = (threadIdx.x / warp_size) % execnum_periter;
            wnum_exec = warp_num / execnum_periter;
            if(eid < warp_num % execnum_periter){
                wnum_exec += 1;
            }
            eid += init_exec + exec_len - remain_execs;
            int wt_exec = wid / execnum_periter;
            exec_id = exec_unit_len * eid;
            exec_unit = exec_units + exec_id;
            
            tid = wt_exec * warp_size + threadIdx.x % warp_size;
            size_t input_size = exec_unit[1];
            scalar_t* output = (scalar_t*)(d_set + exec_unit[2 + input_size] * d_pitch / sizeof(scalar_t)) + d_offset;
            
            scalar_t* input[2], constant[2];
            switch(exec_unit[0]){
                case -1: //'assign'
                    if(exec_unit[2] < 0){
                        constant[0] = constants[-exec_unit[2] - 1];
                    }
                    else{
                        input[0] = (scalar_t*)(d_set + exec_unit[2] * d_pitch / sizeof(scalar_t)) + d_offset;
                    }
                    assign_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);

                    break;
                case 0: //'+'
                    /* */
                    for(int i = 0; i < input_size; ++i){
                        if(exec_unit[2 + i] < 0){
                            constant[i] = constants[-exec_unit[2 + i] - 1];
                        }
                        else{
                            input[i] = (scalar_t*)(d_set + exec_unit[2 + i] * d_pitch / sizeof(scalar_t)) + d_offset;
                        }
                    }
                    //if(exec_unit[2 + input_size] == 102){
                    //    printf("here.........%f, %d, %d, %d, %d, %d\\n", output[1], exec_unit[0], exec_unit[1], exec_unit[2], exec_unit[3], exec_unit[4]);
                    //}
                    //if(blockIdx.x == 0 && threadIdx.x == 0){
                    //    printf("here!!!!!!!!%f, %d, %d, %d, %d, %d\\n", output[1], exec_unit[0], exec_unit[1], exec_unit[2], exec_unit[3], exec_unit[4]);
                    //    //printf("output: %d, %d, %d, %d, %d, %f\\n", exec_unit[2 + input_size], d_offset, wid, exec_len, warp_num, output[1]);
                    //}
                    // if(threadIdx.x == 0 && blockIdx.x == 0){
                    // printf("d_perblock: %ld, wt_exec * warp_size: %ld, wnum_exec * warp_size: %ld, wt_exec: %ld, warp_size: %ld, wnum_exec: %ld\n", d_perblock, wt_exec * warp_size, wnum_exec * warp_size, wt_exec, warp_size, wnum_exec);
                    // }
                    add_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 1: //'-'
                
                    /* */
                    for(int i = 0; i < input_size; ++i){
                        if(exec_unit[2 + i] < 0){
                            constant[i] = constants[-exec_unit[2 + i] - 1];
                        }
                        else{
                            input[i] = (scalar_t*)(d_set + exec_unit[2 + i] * d_pitch / sizeof(scalar_t)) + d_offset;
                        }
                    }
                    
                    sub_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 2: //'*'
                    /* */
                    for(int i = 0; i < input_size; ++i){
                        if(exec_unit[2 + i] < 0){
                            constant[i] = constants[-exec_unit[2 + i] - 1];
                        }
                        else{
                            input[i] = (scalar_t*)(d_set + exec_unit[2 + i] * d_pitch / sizeof(scalar_t)) + d_offset;
                        }
                    }
                    
                    mul_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 3: //'/'
                    /* */
                    for(int i = 0; i < input_size; ++i){
                        if(exec_unit[2 + i] < 0){
                            constant[i] = constants[-exec_unit[2 + i] - 1];
                        }
                        else{
                            input[i] = (scalar_t*)(d_set + exec_unit[2 + i] * d_pitch / sizeof(scalar_t)) + d_offset;
                        }
                    }
                    
                    div_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 4: //'sin'
                    /* */
                    if(exec_unit[2] < 0){
                        constant[0] = constants[-exec_unit[2] - 1];
                    }
                    else{
                        input[0] = (scalar_t*)(d_set + exec_unit[2] * d_pitch / sizeof(scalar_t)) + d_offset;
                    }
                    sin_d(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 5: //'cos'
                    /* */
                    if(exec_unit[2] < 0){
                        constant[0] = constants[-exec_unit[2] - 1];
                    }
                    else{
                        input[0] = (scalar_t*)(d_set + exec_unit[2] * d_pitch / sizeof(scalar_t)) + d_offset;
                    }
                    cos_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 6: //'log'
                    /* */
                    if(exec_unit[2] < 0){
                        constant[0] = constants[-exec_unit[2] - 1];
                    }
                    else{
                        input[0] = (scalar_t*)(d_set + exec_unit[2] * d_pitch / sizeof(scalar_t)) + d_offset;
                    }
                    log_fabs<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 7: //'exp'
                    /* */
                    if(exec_unit[2] < 0){
                        constant[0] = constants[-exec_unit[2] - 1];
                    }
                    else{
                        input[0] = (scalar_t*)(d_set + exec_unit[2] * d_pitch / sizeof(scalar_t)) + d_offset;
                    }
                    exp_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                case 11: // copy
                    for(int i = 0; i < input_size; ++i){
                        if(exec_unit[2 + i] < 0){
                            constant[i] = constants[-exec_unit[2 + i] - 1];
                        }
                        else{
                            input[0] = (scalar_t*)(d_set + exec_unit[2 + i] * d_pitch / sizeof(scalar_t)) + d_offset;
                        }
                    }
                    copy_d<scalar_t>(exec_unit, input, output, constant, d_perblock, wnum_exec * warp_size, tid);
                    break;
                default:
                    printf("operator id out of range..input_id: %d\\n", int(exec_unit[0]));
                    break;
            }
            remain_execs -= execnum_periter;
            __syncthreads();
        }
        init_exec += exec_len;
    }
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //     for(int i = 0; i < d_perblock; ++i){
    //         printf("%f, ", d_set[i]);
    //     }
    //     printf("\n");
    //     printf("----- %d, %ld\n", 2, d_pitch);
    //     for(int i = 0; i < d_perblock; ++i){
    //         printf("%lf,, ", ((scalar_t*)(d_set + 2 * d_pitch / sizeof(scalar_t)) + d_offset)[i]);
    //     }
    //     printf("\n");
    // }
}

#define STREAM_NUM 2

template<typename scalar_t>
void build_np(py::array_t<scalar_t>* d, scalar_t* ptr, std::vector<py::ssize_t> const& o_shape, std::vector<size_t>& init_shape){
    
    for(int i = 2; i < o_shape.size(); ++i){
        init_shape.push_back(o_shape[i]);
    }
    
    std::vector<size_t> strides_r({1});
    for(int i = init_shape.size() - 1; i > 0; --i){
        strides_r.push_back(strides_r.back() * init_shape[i]);
    }

    std::reverse(strides_r.begin(), strides_r.end());

    py::capsule deallocate_buffer_r(ptr, [](void* p) { cudaError_t err = cudaFree(p); 
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));});
    *d = py::array_t<scalar_t>(init_shape, strides_r, ptr, deallocate_buffer_r);
}
#include <ctime>
template<typename scalar_t>
void exec_cpuinput(py::array_t<int> const& execs_list, std::vector<size_t> const& execs_layer,
        py::array_t<scalar_t> const& constants, py::array_t<scalar_t> const& inputs, 
        py::array_t<scalar_t> const& cashes_cpu, 
        std::vector<size_t> const& records_list, const Array<scalar_t>& outputs, const Array<scalar_t>& records,
        std::tuple<size_t, size_t, size_t, size_t, size_t> paras){
    
    CHECK(cudaSetDevice(outputs.device_id));
    const size_t ELEM_SIZE = sizeof(scalar_t);
    // clock_t st = std::clock();
    //paras
    size_t execs_unit_len = std::get<0>(paras);
    size_t xlen = std::get<1>(paras);
    size_t cashs_init_posi = std::get<2>(paras);
    size_t outputs_size = std::get<3>(paras);
    size_t outputs_init_posi = std::get<4>(paras);
    cudaError_t err;
    scalar_t* buf_ptr;
    size_t inputs_len, data_size;

    py::buffer_info buf_input = inputs.request();
    buf_ptr = (scalar_t*)(buf_input.ptr);
    inputs_len = buf_input.shape[0], data_size = buf_input.shape[1];

    py::buffer_info buf_cashes = cashes_cpu.request(), buf_constants = constants.request(), buf_exec_list = execs_list.request();
    const size_t* buf_layers = execs_layer.data();
    size_t execs_len = buf_exec_list.shape[0] / execs_unit_len, cashes_len = buf_cashes.shape[0], constants_len = buf_constants.shape[0], 
     record_num= records_list.size(), layer_num = execs_layer.size(); 
    // printf("execs_len: %d\n", execs_len);
    //variable declaration
    
    // Array outputs((outputs_size) * data_size);// = new scalar_t[(outputs_size) * data_size];
    // err = cudaMalloc(&outputs.ptr, (outputs_size) * data_size * ELEM_SIZE);
    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // clock_t et = std::clock();
    // printf("exec part1-1 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // Array records;// = new scalar_t[(record_num) * data_size];
    // if (record_num > 0){
    //     records.init((record_num) * data_size);
    //     // err = cudaMalloc(&records.ptr, (record_num) * data_size * ELEM_SIZE);
    // }
    // else{
    //     records.init((1) * data_size);
    //     // err = cudaMalloc(&records.ptr, ELEM_SIZE);
    // }
    // // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // mlock(outputs, (outputs_size) * data_size);
    size_t input_pitch;
    // [ ] TODO: iteration time should be compute carefully..
    int min_num_layer = buf_layers[0];
    for(int i = 1; i < layer_num; ++i){
        if (min_num_layer > buf_layers[i]){
            min_num_layer = buf_layers[i];
        }
    }
    size_t pre_mem = constants_len * ELEM_SIZE +
     layer_num * sizeof(size_t) +
     execs_len * execs_unit_len * sizeof(int);
    size_t remain_mem = cuda_mem_available(outputs.device_id) * 9 / 10;
    if (remain_mem - pre_mem < 0) throw std::runtime_error(std::string("No enough CUDA memory to alloc."));
    size_t select_mem = remain_mem > MEM_CAPACITY ? MEM_CAPACITY : remain_mem;
    int iteration = int(std::ceil(double(data_size * ELEM_SIZE * xlen * STREAM_NUM) / select_mem));
    // printf("%d, %d, %d\n", int(data_size * ELEM_SIZE * xlen * STREAM_NUM), int(MEM_CAPACITY), iteration);
    // throw std::runtime_error("assert here..\n");
    size_t d_batch = ceil(data_size / iteration);
    
    int while_max = 5;
    size_t input_ebatch_size = ELEM_SIZE * xlen * STREAM_NUM;
    if(iteration > 1 || pre_mem + d_batch * input_ebatch_size > remain_mem){
        while (true){
            if(pre_mem + d_batch * input_ebatch_size > remain_mem){
                iteration *= 2;
                d_batch = ceil(data_size / iteration);
            }
            else{
                while(pre_mem + d_batch * input_ebatch_size < remain_mem && iteration > 1){
                    iteration -= 1;
                    d_batch = ceil(data_size / iteration);
                }
                if(pre_mem + d_batch * input_ebatch_size >= remain_mem){
                    iteration += 1;
                    d_batch = ceil(data_size / iteration);
                }
                break;
            }
            if(--while_max == 0) throw std::runtime_error(std::string("No enough CUDA memory to alloc."));
        }
    }

    size_t thread_num = 256, block_num = 100, warp_num = thread_num / 32;
    if(warp_num < min_num_layer){
        if(min_num_layer * 32 >= 1024){
            thread_num = 1024;
        }
        else{
            thread_num = (min_num_layer * 32) & (1023);
        }
        warp_num = thread_num / 32;
    }
    if(block_num * 32 * ceil(float(warp_num) / min_num_layer) > d_batch){
        block_num = ceil(double(d_batch) / (32 * ceil(float(warp_num) / min_num_layer)));
    }
    // printf("block_num: %d\n", block_num);
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for(int i = 0; i < STREAM_NUM; ++i){
        cudaStreamCreate(&streams[i]);
    }
    // et = std::clock();
    // printf("exec part1 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // st = std::clock();
    // printf("malloc.....\n");
    ///cuda memory alloc
    int* execs_gpu;
    scalar_t *constants_gpu, *inputs_gpu;
    size_t *execs_layer_gpu;
    err = cudaMalloc((void**)&execs_gpu, execs_len * execs_unit_len * sizeof(int));
    if (err != cudaSuccess) throw std::runtime_error(std::string("execs cudaMalloc:") + cudaGetErrorString(err));
    err = cudaMalloc((void**)&constants_gpu, constants_len * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(std::string("constants cudaMalloc:") + cudaGetErrorString(err));
    err = cudaMalloc((void**)&execs_layer_gpu, layer_num * sizeof(size_t));
    if (err != cudaSuccess) throw std::runtime_error(std::string("execs_infos cudaMalloc:") + cudaGetErrorString(err));
    err = cudaMallocPitch((void**)&inputs_gpu, &input_pitch, d_batch * ELEM_SIZE, xlen * STREAM_NUM);
    if (err != cudaSuccess) throw std::runtime_error(std::string("inputs cudaMalloc:") + cudaGetErrorString(err));
    // err = cudaMalloc((void**)&inputs_gpu, d_batch * ELEM_SIZE * xlen * STREAM_NUM);
    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // printf("malloc finish.....\n");
    ///cuda memcpy
    err = cudaMemcpy(execs_gpu, buf_exec_list.ptr, execs_len * execs_unit_len * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpy(constants_gpu, buf_constants.ptr, constants_len * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpy(execs_layer_gpu, buf_layers, layer_num * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    cudaMemcpy2DAsync(inputs_gpu, input_pitch, buf_ptr, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, inputs_len, cudaMemcpyHostToDevice, streams[0]);
    
    cudaMemcpy2DAsync(inputs_gpu + input_pitch * cashs_init_posi / sizeof(scalar_t), input_pitch, buf_cashes.ptr, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, cashes_len, cudaMemcpyHostToDevice, streams[0]);
    
    // cudaDeviceSynchronize();
    // et = std::clock();
    // printf("exec part2 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // cudaMemcpyAsync(inputs_gpu, buf_ptr, d_batch * ELEM_SIZE * inputs_len, cudaMemcpyHostToDevice, streams[0]);
    // cudaMemcpyAsync(inputs_gpu + d_batch * cashs_init_posi, buf_cashes.ptr, d_batch * ELEM_SIZE * cashes_len, cudaMemcpyHostToDevice, streams[0]);
    // st = std::clock();
    // printf("0 iteration: %d\n", iteration);
    for(int i = 0; i < iteration; ++i){
        // printf("iteration: %d, %ld\n", i, iteration);
        size_t offset = (i % STREAM_NUM) * xlen;
        if (i + 1 < iteration){
            // size_t inputs_gpu_init = ((i + 1) % STREAM_NUM) * input_pitch * xlen / sizeof(scalar_t);
            size_t inputs_gpu_init = ((i + 1) % STREAM_NUM) * d_batch * xlen;
            ///memcpy next data in advance, to overlay the transfer and computation.
            
            cudaMemcpy2DAsync(inputs_gpu + inputs_gpu_init, input_pitch, buf_ptr + d_batch * i, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, inputs_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
            
            cudaMemcpy2DAsync(inputs_gpu + inputs_gpu_init + cashs_init_posi * (input_pitch / sizeof(scalar_t)), input_pitch, buf_cashes.ptr + d_batch * i, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, cashes_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
            // cudaMemcpyAsync(inputs_gpu + inputs_gpu_init, buf_ptr + d_batch * inputs_len * i, d_batch * ELEM_SIZE * inputs_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
            // cudaMemcpyAsync(inputs_gpu + inputs_gpu_init + cashs_init_posi * d_batch, buf_cashes.ptr + d_batch * cashes_len * i, d_batch * ELEM_SIZE * cashes_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
        }
        // size_t inputs_gpu_init = (i % STREAM_NUM) * input_pitch * xlen / sizeof(scalar_t);
        size_t inputs_gpu_init = (i % STREAM_NUM) * d_batch * xlen;
        // printf("0 iteration: %d, %d\n", iteration, i);
        execution_GPU<scalar_t><<<block_num, thread_num, 0, streams[i % STREAM_NUM]>>>(execs_gpu, execs_unit_len, execs_layer_gpu, layer_num, inputs_gpu, input_pitch, offset, d_batch, constants_gpu); 
        // cudaMemcpyAsync(outputs + d_batch * outputs_size * i, inputs_gpu + inputs_gpu_init + outputs_init_posi * d_batch, d_batch * ELEM_SIZE * outputs_size, cudaMemcpyDeviceToHost, streams[i % STREAM_NUM]);
        // for(int j = 0; j < record_num; ++j){
        //     cudaMemcpyAsync(records + d_batch * (record_num * i + j), inputs_gpu + inputs_gpu_init + records_list[j] * d_batch, d_batch * ELEM_SIZE, cudaMemcpyDeviceToHost, streams[i % STREAM_NUM]);
        // }
        
        cudaMemcpy2DAsync(outputs.ptr + d_batch * i, data_size * ELEM_SIZE, inputs_gpu + inputs_gpu_init + outputs_init_posi * (input_pitch / sizeof(scalar_t)), input_pitch, d_batch * ELEM_SIZE, outputs_size, cudaMemcpyDeviceToDevice, streams[i % STREAM_NUM]);
        for(int j = 0; j < record_num; ++j){
            cudaMemcpy2DAsync(records.ptr + d_batch * i + data_size * j, data_size * ELEM_SIZE, inputs_gpu + inputs_gpu_init + records_list[j] * (input_pitch / sizeof(scalar_t)), input_pitch, d_batch * ELEM_SIZE, 1, cudaMemcpyDeviceToDevice, streams[i % STREAM_NUM]);
        }
    }
    // printf("iteration: %d, %ld\n", iteration, iteration);
    // cudaDeviceSynchronize();
    // et = std::clock();
    // printf("exec part3 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // st = std::clock();

    cudaFree(execs_gpu);
    cudaFree(constants_gpu);
    cudaFree(execs_layer_gpu);
    // scalar_t* inputs_gpu = cudaMalloc(inputs_len * ELEM_SIZE);
    cudaFree(inputs_gpu);
    

    // printf("4here..\n");
    for (int i = 0; i < STREAM_NUM; ++i){
        cudaStreamDestroy(streams[i]);
    }
    cudaError_t err_l = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

    // std::vector<size_t> shape({outputs_size, data_size});
    // py::array_t<scalar_t> outputs_np;
    // build_np(&outputs_np, outputs, buf_input.shape, shape);
    
    
    // std::vector<size_t> shape_r({record_num, data_size});
    // py::array_t<scalar_t> records_np;
    // build_np(&records_np, records, buf_input.shape, shape_r);
    
    // et = std::clock();
    // printf("exec part4 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // munlock(outputs, (outputs_size) * data_size);
    // err_l = cudaGetLastError();
    // printf("Error_1: %s\n", cudaGetErrorString(err_l));

    // return std::tuple<Array, Array>(outputs, records);
}

template<typename scalar_t>
void exec_gpuinput(py::array_t<int> const& execs_list, std::vector<size_t> const& execs_layer,
        py::array_t<scalar_t> const& constants, InputCuda const& inputs, 
        py::array_t<scalar_t> const& cashes_cpu, 
        std::vector<size_t> const& records_list, const Array<scalar_t>& outputs, const Array<scalar_t>& records,
        std::tuple<size_t, size_t, size_t, size_t, size_t> paras){
    
    CHECK(cudaSetDevice(outputs.device_id));
    const size_t ELEM_SIZE = sizeof(scalar_t);
    // clock_t st = std::clock();
    //paras
    size_t execs_unit_len = std::get<0>(paras);
    size_t xlen = std::get<1>(paras);
    size_t cashs_init_posi = std::get<2>(paras);
    size_t outputs_size = std::get<3>(paras);
    size_t outputs_init_posi = std::get<4>(paras);
    cudaError_t err;
    scalar_t* buf_ptr;
    size_t inputs_len, data_size;
    buf_ptr = (scalar_t*)(reinterpret_cast<scalar_t*>(inputs.ptr) + inputs.offset);
    inputs_len = inputs.shape[0], data_size = inputs.shape[1];
    py::buffer_info buf_cashes = cashes_cpu.request(), buf_constants = constants.request(), buf_exec_list = execs_list.request();
    const size_t* buf_layers = execs_layer.data();
    size_t execs_len = buf_exec_list.shape[0] / execs_unit_len, cashes_len = buf_cashes.shape[0], constants_len = buf_constants.shape[0], 
     record_num= records_list.size(), layer_num = execs_layer.size(); 
    // printf("execs_len: %d\n", execs_len);
    //variable declaration
    
    // Array outputs((outputs_size) * data_size);// = new scalar_t[(outputs_size) * data_size];
    // err = cudaMalloc(&outputs.ptr, (outputs_size) * data_size * ELEM_SIZE);
    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // clock_t et = std::clock();
    // printf("exec part1-1 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // Array records;// = new scalar_t[(record_num) * data_size];
    // if (record_num > 0){
    //     records.init((record_num) * data_size);
    //     // err = cudaMalloc(&records.ptr, (record_num) * data_size * ELEM_SIZE);
    // }
    // else{
    //     records.init((1) * data_size);
    //     // err = cudaMalloc(&records.ptr, ELEM_SIZE);
    // }
    // // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // mlock(outputs, (outputs_size) * data_size);
    size_t input_pitch;
    // [ ] TODO: iteration time should be compute carefully..
    int min_num_layer = buf_layers[0];
    for(int i = 1; i < layer_num; ++i){
        if (min_num_layer > buf_layers[i]){
            min_num_layer = buf_layers[i];
        }
    }
    size_t pre_mem = constants_len * ELEM_SIZE +
     layer_num * sizeof(size_t) +
     execs_len * execs_unit_len * sizeof(int);
    size_t remain_mem = cuda_mem_available(outputs.device_id) * 9 / 10;
    if (remain_mem - pre_mem < 0) throw std::runtime_error(std::string("No enough CUDA memory to alloc."));
    size_t select_mem = remain_mem > MEM_CAPACITY ? MEM_CAPACITY : remain_mem;
    int iteration = int(std::ceil(double(data_size * ELEM_SIZE * xlen * STREAM_NUM) / select_mem));
    // printf("%d, %d, %d\n", int(data_size * ELEM_SIZE * xlen * STREAM_NUM), int(select_mem), iteration);
    // throw std::runtime_error("assert here..\n");
    size_t d_batch = ceil(data_size / iteration);

    int while_max = 5;
    size_t input_ebatch_size = ELEM_SIZE * xlen * STREAM_NUM;
    if(iteration > 1 || pre_mem + d_batch * input_ebatch_size > remain_mem){
        while (true){
            if(pre_mem + d_batch * input_ebatch_size > remain_mem){
                iteration *= 2;
                d_batch = ceil(data_size / iteration);
            }
            else{
                while(pre_mem + d_batch * input_ebatch_size < remain_mem && iteration > 1){
                    iteration -= 1;
                    d_batch = ceil(data_size / iteration);
                }
                if(pre_mem + d_batch * input_ebatch_size >= remain_mem){
                    iteration += 1;
                    d_batch = ceil(data_size / iteration);
                }
                break;
            }
            if(--while_max == 0) throw std::runtime_error(std::string("No enough CUDA memory to alloc."));
        }
    }
    
    size_t thread_num = 512, block_num = 100, warp_num = thread_num / 32;
    if(warp_num < min_num_layer){
        if(min_num_layer * 32 >= 1024){
            thread_num = 1024;
        }
        else{
            thread_num = (min_num_layer * 32) & (1023);
        }
        warp_num = thread_num / 32;
    }
    if(block_num * 32 * ceil(float(warp_num) / min_num_layer) > d_batch){
        block_num = ceil(double(d_batch) / (32 * ceil(float(warp_num) / min_num_layer)));
    }
    // printf("block_num: %d\n", block_num);
    cudaStream_t* streams = new cudaStream_t[STREAM_NUM];
    for(int i = 0; i < STREAM_NUM; ++i){
        cudaStreamCreate(&streams[i]);
    }
    // et = std::clock();
    // printf("exec part1 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // st = std::clock();
    // printf("malloc.....\n");
    ///cuda memory alloc
    int* execs_gpu;
    scalar_t *constants_gpu, *inputs_gpu;
    size_t *execs_layer_gpu;
    err = cudaMalloc((void**)&execs_gpu, execs_len * execs_unit_len * sizeof(int));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMalloc((void**)&constants_gpu, constants_len * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMalloc((void**)&execs_layer_gpu, layer_num * sizeof(size_t));
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    err = cudaMallocPitch((void**)&inputs_gpu, &input_pitch, d_batch * ELEM_SIZE, xlen * STREAM_NUM);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    // err = cudaMalloc((void**)&inputs_gpu, d_batch * ELEM_SIZE * xlen * STREAM_NUM);
    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    // printf("malloc finish.....\n");
    ///cuda memcpy
    err = cudaMemcpy(execs_gpu, buf_exec_list.ptr, execs_len * execs_unit_len * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpy(constants_gpu, buf_constants.ptr, constants_len * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    err = cudaMemcpy(execs_layer_gpu, buf_layers, layer_num * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    
    cudaMemcpy2DAsync(inputs_gpu, input_pitch, buf_ptr, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, inputs_len, cudaMemcpyDeviceToDevice, streams[0]);
    cudaMemcpy2DAsync(inputs_gpu + input_pitch * cashs_init_posi / sizeof(scalar_t), input_pitch, buf_cashes.ptr, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, cashes_len, cudaMemcpyHostToDevice, streams[0]);
    
    // cudaDeviceSynchronize();
    // et = std::clock();
    // printf("exec part2 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // cudaMemcpyAsync(inputs_gpu, buf_ptr, d_batch * ELEM_SIZE * inputs_len, cudaMemcpyHostToDevice, streams[0]);
    // cudaMemcpyAsync(inputs_gpu + d_batch * cashs_init_posi, buf_cashes.ptr, d_batch * ELEM_SIZE * cashes_len, cudaMemcpyHostToDevice, streams[0]);
    // st = std::clock();
    // printf("1 iteration: %d\n", iteration);
    for(int i = 0; i < iteration; ++i){
        // printf("iteration: %d, %ld\n", i, iteration);
        size_t offset = (i % STREAM_NUM) * xlen;
        if (i + 1 < iteration){
            // size_t inputs_gpu_init = ((i + 1) % STREAM_NUM) * input_pitch * xlen / sizeof(scalar_t);
            size_t inputs_gpu_init = ((i + 1) % STREAM_NUM) * d_batch * xlen;
            ///memcpy next data in advance, to overlay the transfer and computation.
              
            cudaMemcpy2DAsync(inputs_gpu + inputs_gpu_init, input_pitch, buf_ptr + d_batch * i, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, inputs_len, cudaMemcpyDeviceToDevice, streams[(i + 1) % STREAM_NUM]);
            cudaMemcpy2DAsync(inputs_gpu + inputs_gpu_init + cashs_init_posi * (input_pitch / sizeof(scalar_t)), input_pitch, buf_cashes.ptr + d_batch * i, data_size * ELEM_SIZE, d_batch * ELEM_SIZE, cashes_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
            // cudaMemcpyAsync(inputs_gpu + inputs_gpu_init, buf_ptr + d_batch * inputs_len * i, d_batch * ELEM_SIZE * inputs_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
            // cudaMemcpyAsync(inputs_gpu + inputs_gpu_init + cashs_init_posi * d_batch, buf_cashes.ptr + d_batch * cashes_len * i, d_batch * ELEM_SIZE * cashes_len, cudaMemcpyHostToDevice, streams[(i + 1) % STREAM_NUM]);
        }
        // size_t inputs_gpu_init = (i % STREAM_NUM) * input_pitch * xlen / sizeof(scalar_t);
        size_t inputs_gpu_init = (i % STREAM_NUM) * d_batch * xlen;

        execution_GPU<scalar_t><<<block_num, thread_num, 0, streams[i % STREAM_NUM]>>>(execs_gpu, execs_unit_len, execs_layer_gpu, layer_num, inputs_gpu, input_pitch, offset, d_batch, constants_gpu); 
        // cudaMemcpyAsync(outputs + d_batch * outputs_size * i, inputs_gpu + inputs_gpu_init + outputs_init_posi * d_batch, d_batch * ELEM_SIZE * outputs_size, cudaMemcpyDeviceToHost, streams[i % STREAM_NUM]);
        // for(int j = 0; j < record_num; ++j){
        //     cudaMemcpyAsync(records + d_batch * (record_num * i + j), inputs_gpu + inputs_gpu_init + records_list[j] * d_batch, d_batch * ELEM_SIZE, cudaMemcpyDeviceToHost, streams[i % STREAM_NUM]);
        // }
        
        cudaMemcpy2DAsync(outputs.ptr + d_batch * i, data_size * ELEM_SIZE, inputs_gpu + inputs_gpu_init + outputs_init_posi * (input_pitch / sizeof(scalar_t)), input_pitch, d_batch * ELEM_SIZE, outputs_size, cudaMemcpyDeviceToDevice, streams[i % STREAM_NUM]);
        for(int j = 0; j < record_num; ++j){
            cudaMemcpy2DAsync(records.ptr + d_batch * i + data_size * j, data_size * ELEM_SIZE, inputs_gpu + inputs_gpu_init + records_list[j] * (input_pitch / sizeof(scalar_t)), input_pitch, d_batch * ELEM_SIZE, 1, cudaMemcpyDeviceToDevice, streams[i % STREAM_NUM]);
        }
    }
    // printf("iteration: %d, %ld\n", iteration, iteration);
    // cudaDeviceSynchronize();
    // et = std::clock();
    // printf("exec part3 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // st = std::clock();

    cudaFree(execs_gpu);
    cudaFree(constants_gpu);
    cudaFree(execs_layer_gpu);
    // test<scalar_t><<<1, 32>>>(outputs.ptr, 0, 0);
    // printf("outputs_init_posi: %d, %d\n", outputs_init_posi, (input_pitch / sizeof(scalar_t)));
    // test<scalar_t><<<1, 32>>>(inputs_gpu + 0 + outputs_init_posi * (input_pitch / sizeof(scalar_t)), 0, 0);
    // scalar_t* inputs_gpu = cudaMalloc(inputs_len * ELEM_SIZE);
    cudaFree(inputs_gpu);

    // printf("4here..\n");
    for (int i = 0; i < STREAM_NUM; ++i){
        cudaStreamDestroy(streams[i]);
    }
    cudaError_t err_l = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

    // std::vector<size_t> shape({outputs_size, data_size});
    // py::array_t<scalar_t> outputs_np;
    // build_np(&outputs_np, outputs, buf_input.shape, shape);
    
    
    // std::vector<size_t> shape_r({record_num, data_size});
    // py::array_t<scalar_t> records_np;
    // build_np(&records_np, records, buf_input.shape, shape_r);
    
    // et = std::clock();
    // printf("exec part4 time: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    // munlock(outputs, (outputs_size) * data_size);
    // err_l = cudaGetLastError();
    // printf("Error_1: %s\n", cudaGetErrorString(err_l));

    // return std::tuple<Array, Array>(outputs, records);
}

}




template<typename scalar_t>
void TEMPLATE_BIND_FUNCS(py::module& m){
    using namespace pygp_exec;

    m.def("exec_cpuinput", &exec_cpuinput<scalar_t>);
    m.def("exec_gpuinput", &exec_gpuinput<scalar_t>);
    
}

PYBIND11_MODULE(executor, m){

    py::class_<InputCuda>(m, "InputCuda")
        .def(py::init<size_t, int, std::vector<size_t>&>(), py::return_value_policy::take_ownership);

    TEMPLATE_BIND_FUNCS<int8_t>(m);
    TEMPLATE_BIND_FUNCS<int16_t>(m);
    TEMPLATE_BIND_FUNCS<int32_t>(m);
    TEMPLATE_BIND_FUNCS<int64_t>(m);
    TEMPLATE_BIND_FUNCS<float>(m);
    TEMPLATE_BIND_FUNCS<double>(m);
}
