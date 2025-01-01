#include <cuda_runtime.h>



#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#include "cuda.h"
#include<Array.h>
#include <cublas_v2.h>
#include<assert.h>

// #include "npp.h"
// #include "nppi.h"
// #include "npps.h"


namespace py = pybind11;

cublasHandle_t cublas_handle;
void handle_create(){
    cublasCreate_v2(&cublas_handle);
}

void streams_create(const std::vector<int>& devices){
    for(int i = 0; i < devices.size(); ++i){
        if(streams.find(devices[i]) == streams.end()){
            if (is_available(devices[i])){
                CHECK(cudaSetDevice(devices[i]));
                streams[devices[i]] = new cudaStream_t[STREAM_NUM_NDARRAY];
                for(int k = 0; k < STREAM_NUM_NDARRAY; ++k){
                    cudaStreamCreate(&streams[devices[i]][k]);
                }
            }
        }
    }
}

template<typename scalar_t, typename sscalar_t>
scalar_t* cpy_gpus(const Array<scalar_t>& in, Array<sscalar_t>& out, Array<scalar_t>& new_in){
    if(in.device_id == out.device_id){
        CHECK(cudaSetDevice(out.device_id));
        
        ewise_async_1op(in, out);
        return in.ptr;
    }
    else{
        // printf("we in here???\n");
        new_in.init(in.size, out.device_id);
        if(in.stream_id != -1) cudaStreamWaitEvent(streams[in.device_id][in.stream_id], in.event_sign, 0);
        cudaMemcpyPeerAsync(new_in.ptr, new_in.device_id, in.ptr, in.device_id, sizeof(scalar_t) * in.size);
        CHECK(cudaSetDevice(out.device_id));
        ewise_async_1op(new_in, out);
        return new_in.ptr;
    }
}

template<typename scalar_t, typename sscalar_t>
Array<scalar_t>& cpy_gpus(const Array<scalar_t>& in, const Array<sscalar_t>& out){
    Array<scalar_t> new_in(in.size, out.device_id);
    cudaMemcpyPeerAsync(new_in.ptr, new_in.device_id, in.ptr, in.device_id, sizeof(scalar_t) * in.size);
    CHECK(cudaSetDevice(out.device_id));
    return new_in;
}

#define INIT_THREAD_NUM 256
#define CASH_NUM 10

#include <stdio.h>

 
void get_properties()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
 
    if (deviceCount == 0)
    {
       printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
       printf("Detected %d CUDA Capable device(s)\n", deviceCount);
       printf("\n");
    }
 
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    for(dev = 0; dev < deviceCount; dev++)
    {
       CHECK(cudaSetDevice(dev));
       cudaDeviceProp deviceProp;
       CHECK(cudaGetDeviceProperties(&deviceProp, dev));
       printf("Device %d: \"%s\"\n", dev, deviceProp.name);
 
       cudaDriverGetVersion(&driverVersion);
       cudaRuntimeGetVersion(&runtimeVersion);
       printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
              driverVersion / 1000, (driverVersion % 100) / 10,
              runtimeVersion / 1000, (runtimeVersion % 100) / 10);
       printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
              deviceProp.major, deviceProp.minor);
       printf("  Total amount of global memory:                 %.2f GBytes (%llu "
              "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
              (unsigned long long)deviceProp.totalGlobalMem);
       printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
              "GHz)\n", deviceProp.clockRate * 1e-3f,
              deviceProp.clockRate * 1e-6f);
       printf("  Memory Clock rate:                             %.0f Mhz\n",
              deviceProp.memoryClockRate * 1e-3f);
       printf("  Memory Bus Width:                              %d-bit\n",
              deviceProp.memoryBusWidth);
 
       if (deviceProp.l2CacheSize)
       {
              printf("  L2 Cache Size:                                 %d bytes\n",
                     deviceProp.l2CacheSize);
       }
 
       printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
              "2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
              deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
              deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
              deviceProp.maxTexture3D[2]);
       printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
              "2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
              deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
              deviceProp.maxTexture2DLayered[1],
              deviceProp.maxTexture2DLayered[2]);
       printf("  Total amount of constant memory:               %lu bytes\n",
              deviceProp.totalConstMem);
       printf("  Total amount of shared memory per block:       %lu bytes\n",
              deviceProp.sharedMemPerBlock);
       printf("  Total number of registers available per block: %d\n",
              deviceProp.regsPerBlock);
       printf("  Warp size:                                     %d\n",
              deviceProp.warpSize);
       printf("  Maximum number of threads per multiprocessor:  %d\n",
              deviceProp.maxThreadsPerMultiProcessor);
       printf("  Number of multiprocessor:                      %d\n",
              deviceProp.multiProcessorCount);
       printf("  Maximum number of threads per block:           %d\n",
              deviceProp.maxThreadsPerBlock);
       printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
              deviceProp.maxThreadsDim[0],
              deviceProp.maxThreadsDim[1],
              deviceProp.maxThreadsDim[2]);
       printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
              deviceProp.maxGridSize[0],
              deviceProp.maxGridSize[1],
              deviceProp.maxGridSize[2]);
       printf("  Maximum memory pitch:                          %lu bytes\n",
              deviceProp.memPitch);
       printf("\n");
    }

    // printf("==============TEST the support for memory pools===================\n");
    // int deviceSupportsMemoryPools = 0;  
    // int poolSupportedHandleTypes = 0;  
    // cudaDriverGetVersion(&driverVersion);  
    // if (driverVersion >= 11020) {  
    //     cudaDeviceGetAttribute(&deviceSupportsMemoryPools,  
    //                             cudaDevAttrMemoryPoolsSupported, device);  
    // }  
    // if (deviceSupportsMemoryPools != 0) {  
    //     // `device` supports the Stream Ordered Memory Allocator  
    // }  
    
    // if (driverVersion >= 11030) {  
    //     cudaDeviceGetAttribute(&poolSupportedHandleTypes,  
    //             cudaDevAttrMemoryPoolSupportedHandleTypes, device);  
    // }  
    // if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {  
    //     // Pools on the specified device can be created with posix file descriptor-based IPC  
    // }  
    // printf("==================================================================\n");
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

struct CudaDims {
  dim3 block, grid;
};

#define BASE_THREAD_NUM 256
CudaDims CudaOneDim(size_t size) {
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}


CudaVec VecToCuda(const std::vector<int32_t> &x)
{
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}


template<typename scalar_t, typename sscalar_t>
size_t ewise_async_1op(const Array<scalar_t>& a, Array<sscalar_t>& out){
    size_t block_num = (out.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if (a.stream_id == -1){
        if(out.stream_id == -1) out.stream_id = rand() % STREAM_NUM_NDARRAY;
    }
    else{
        out.stream_id = a.stream_id;
    }
    return block_num;
}

///////////////////////////////////////////////////////////////////////
// bind the computation with corresponding stream
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t>
size_t ewise_async_2op(const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>& out){
    size_t block_num = (out.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if(a.stream_id == -1 && b.stream_id == -1){
        if(out.stream_id == -1) out.stream_id = rand() % STREAM_NUM_NDARRAY;
    }
    else if(out.stream_id != -1){
        cudaStreamWaitEvent(streams[out.device_id][out.stream_id], a.event_sign, 0);
        cudaStreamWaitEvent(streams[out.device_id][out.stream_id], b.event_sign, 0);
    }
    else{
        if(a.stream_id != -1){
            out.stream_id = a.stream_id;
            cudaStreamWaitEvent(streams[out.device_id][out.stream_id], b.event_sign, 0);
        }
        else{
            out.stream_id = b.stream_id;
        }
    }
    return block_num;
}

namespace gpu{
namespace pygp_tensor{
    

template<typename scalar_t>
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact Array<scalar_t> out.
   * 
   * Args:
   *   a: CUDA pointer to a Array<scalar_t>
   *   out: CUDA point to out Array<scalar_t>
   *   size: size of out Array<scalar_t>
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out Array<scalar_t>
   *   offset: offset of out Array<scalar_t>
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t gid_a = 0;
    size_t tmp = gid;
    for (int i=shape.size-1; i>=0; i--) {
      size_t idx = tmp % shape.data[i];
      tmp /= shape.data[i];
      gid_a += idx * strides.data[i];
    }
    out[gid] = a[gid_a + offset];
  }
}

template<typename scalar_t>
void Compact(const Array<scalar_t>& a_handle, Array<scalar_t>& out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
    /**
     * Compact an Array<scalar_t> in memory. 
     * 
     * Args:
     *   a: non-compact represntation of the Array<scalar_t>, given as input
     *   out: compact version of the Array<scalar_t> to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *a* Array<scalar_t> (not out, which has compact strides)
     *   offset: offset of the *a* Array<scalar_t> (not out, which has zero offset, being compact)
     */
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);
    
    CudaDims dim = CudaOneDim(out.size);
    CompactKernel<scalar_t><<<dim.grid, dim.block, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, out.size, VecToCuda(shape),
                                            VecToCuda(strides), offset);
    
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t>
__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
  CudaVec strides, size_t offset) {
/**
* The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
* non-compact input a, to the corresponding item (at location gid) in the compact Array<scalar_t> out.
* 
* Args:
*   a: CUDA pointer to a Array<scalar_t>
*   out: CUDA point to out Array<scalar_t>
*   size: size of out Array<scalar_t>
*   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
*   strides: vector of strides of out Array<scalar_t>
*   offset: offset of out Array<scalar_t>
*/
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t gid_out = 0;
    size_t tmp = gid;
    for (int i=shape.size-1; i>=0; i--) {
      size_t idx = tmp % shape.data[i];
      tmp /= shape.data[i];
      gid_out += idx * strides.data[i];
    }
    out[gid_out + offset] = a[gid];
  }
}

template<typename scalar_t>
void EwiseSetitem(const Array<scalar_t>& a_handle, Array<scalar_t>& out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) Array<scalar_t> using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ Array<scalar_t> whose items will be written to out
   *   out: non-compact Array<scalar_t> whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* Array<scalar_t> (not a, which has compact strides)
   *   offset: offset of the *out* Array<scalar_t> (not a, which has zero offset, being compact)
   */
  // NOTE: a.size, not out->size. Because a is compacted while out is not compacted.

    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);
                  
    CudaDims dim = CudaOneDim(a.size);
    EwiseSetitemKernel<scalar_t><<<1, dim.block, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, a.size, VecToCuda(shape),
                                                VecToCuda(strides), offset);
                       
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t>
__global__ void ConcatenateKernel(scalar_t* out, scalar_t* a_handle, int32_t pre_size, int32_t post_len, int32_t array_posi, int32_t offset, int32_t offset_a){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    for(int i = gid; i < pre_size * post_len; i += t_n){
        size_t pre_id = int(i / post_len), post_posi = i % post_len;
        out[pre_id * offset + array_posi + post_posi] = a[post_posi + pre_id * post_len];
    }
}

template<typename scalar_t>
__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
  CudaVec strides, size_t offset) {
/**
* The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
* non-compact input a, to the corresponding item (at location gid) in the compact Array<scalar_t> out.
* 
* Args:
*   a: CUDA pointer to a Array<scalar_t>
*   out: CUDA point to out Array<scalar_t>
*   size: size of out Array<scalar_t>
*   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
*   strides: vector of strides of out Array<scalar_t>
*   offset: offset of out Array<scalar_t>
*/
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    size_t gid_out = 0;
    size_t tmp = gid;
    for (int i=shape.size-1; i>=0; i--) {
      size_t idx = tmp % shape.data[i];
      tmp /= shape.data[i];
      gid_out += idx * strides.data[i];
    }
    out[gid_out + offset] = val;
  }
}

template<typename scalar_t>
void Concatenate(Array<scalar_t>& out, const Array<scalar_t>& a_handle, int32_t pre_size, int32_t post_len, int32_t array_posi, int32_t offset, int32_t offset_a){
    
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);
    CudaDims dim = CudaOneDim(pre_size * post_len);
    ConcatenateKernel<scalar_t><<<1, dim.block, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, a_ptr, pre_size, post_len, array_posi, offset, offset_a);

    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t>
void ScalarSetitem(size_t size, scalar_t val, Array<scalar_t>& out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) Array<scalar_t>
   * 
   * Args:
   *   size: number of elements to write in out Array<scalar_t> (note that this will note be the same as
   *         out.size, because out is a non-compact subset Array<scalar_t>);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact Array<scalar_t> whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out Array<scalar_t>
   *   offset: offset of the out Array<scalar_t>
   */
  
    
    if (out.stream_id == -1){
        out.stream_id = rand() % STREAM_NUM_NDARRAY;
    }
                  
    CudaDims dim = CudaOneDim(out.size);
    ScalarSetitemKernel<scalar_t><<<1, dim.block, 0, streams[out.device_id][out.stream_id]>>>(val, out.ptr, out.size, VecToCuda(shape),
                                                VecToCuda(strides), offset);

    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

///////////////////////////////////////////////////////////////////////
// ewise gpu operator
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t, typename fscalar_t>
__device__ void _ewise_where_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, fscalar_t* cash_num_c, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        if (cash_num_a[ptr]){
            out[ptr] = cash_num_b[ptr];
        }
        else{
            out[ptr] = cash_num_a[ptr];
        }
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_add_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] + cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_sub_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] - cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_mul_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] * cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_div_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] / cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_pow_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = pow(cash_num_a[ptr], cash_num_b[ptr]);
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_le_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] <= cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_lt_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] < cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_ge_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] >= cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_gt_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] > cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_ne_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] != cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _ewise_eq_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* cash_num_b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cash_num_a[ptr] == cash_num_b[ptr];
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_sin_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = sin(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_cos_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = cos(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_tan_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = tan(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_sqrt_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = sqrt(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_arcsin_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = asin(double(cash_num_a[ptr]));

    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_arccos_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = acos(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_arctan_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = atan(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_sign_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = -(cash_num_a[ptr] < 0.f) + (cash_num_a[ptr] > 0.f);
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_abs_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = fabs(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_exp_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = exp(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_neg_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = -cash_num_a[ptr];
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_ceil_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = ceil(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_floor_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = floor(double(cash_num_a[ptr]));
    }
}

template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_loge_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = logf(double(cash_num_a[ptr]));
    }
}
template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_log10_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = log10f(double(cash_num_a[ptr]));
    }
}
template<typename scalar_t, typename sscalar_t>
__device__ void _ewise_log2_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        size_t ptr = c_id + i;
        out[ptr] = log2f(double(cash_num_a[ptr]));
    }
}

// __device__ void _ewise_eq_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, scalar_t* cash_num_b, scalar_t* out){
//     for(int i = 0; i < remain_size; ++i){
//         if (cash_num_b[i] != cash_num_a[i]){
//             out = false;
//             break;
//         }
//     }
//     __threadfence();
//     __syncthreads();
// }

template<typename scalar_t, typename sscalar_t>
__global__ void ewise_assign_global(scalar_t* a_handle, sscalar_t val, int size, int offset){
    size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset);
    
    while(c_id < size){
        a[c_id] = val;
        c_id += t_n;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t, typename fscalar_t>
__global__ void ewise_compute_gpu_3op(scalar_t* a_handle, sscalar_t* b_handle, tscalar_t* c_handle, fscalar_t* out, size_t size, int32_t offset_a, int32_t offset_b, int32_t offset_c, size_t type){
    size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    sscalar_t *b = (sscalar_t*)(b_handle + offset_b);
    tscalar_t *c = (tscalar_t*)(c_handle + offset_c);

    while(c_id < size){
        size_t remain_size = (size - c_id);
        /////////////////////////////////////////////////////
        // divide the transfer and computation process, to
        // avoid the time cost of data transfer best
        ////////////////////////////////////////////////////
        size_t remain_time = CASH_NUM * t_n;
        if (remain_size < CASH_NUM * t_n){
            remain_time = remain_size;
        }
        switch(type){
            case 0://'where'
                _ewise_where_gpu<scalar_t, sscalar_t, tscalar_t, fscalar_t>(remain_time, c_id, t_n, a, b, c, out);
                break;
            default:
                printf("Warning: the input type out of available type");
        }
        c_id += remain_time * t_n;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__global__ void ewise_compute_gpu_2op(scalar_t* a_handle, sscalar_t* b_handle, tscalar_t* out, size_t size, int32_t offset_a, int32_t offset_b, size_t type){
    size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    sscalar_t *b = (sscalar_t*)(b_handle + offset_b);

    while(c_id < size){
        size_t remain_size = (size - c_id);
        /////////////////////////////////////////////////////
        // divide the transfer and computation process, to
        // avoid the time cost of data transfer best
        ////////////////////////////////////////////////////
        size_t remain_time = CASH_NUM * t_n;
        if (remain_size < CASH_NUM * t_n){
            remain_time = remain_size;
        }
        switch(type){
            case 0://'+'
                _ewise_add_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 1://'-'
                _ewise_sub_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 2://'*'
                _ewise_mul_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 3://'/'
                _ewise_div_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 4://'pow'
                _ewise_pow_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 5://'le'
                _ewise_le_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 6://'lt'
                _ewise_lt_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 7://'ge'
                _ewise_ge_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 8://'gt'
                _ewise_gt_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 9://'ne'
                _ewise_ne_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 10://'eq'
                _ewise_eq_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            default:
                printf("Warning: the input type out of available type");
        }
        c_id += remain_time * t_n;
    }
}

template<typename scalar_t, typename sscalar_t>
__global__ void ewise_compute_gpu_1op(scalar_t* a_handle, sscalar_t* out, size_t size, int32_t offset_a, size_t type){
    size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    while(c_id < size){
        size_t remain_size = size - c_id;
        /////////////////////////////////////////////////////
        // divide the transfer and computation process, to
        // avoid the time cost of data transfer best
        ////////////////////////////////////////////////////
        size_t remain_time = CASH_NUM * t_n;
        if (remain_size < CASH_NUM * t_n){
            remain_time = remain_size;
        }
        switch(type){
            case 0://'sin'
                _ewise_sin_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 1://'cos'
                _ewise_cos_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 2://'tan'
                _ewise_tan_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 3://'sqrt'
                _ewise_sqrt_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 4://'arcsin'
                _ewise_arcsin_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 5://'arccos'
                _ewise_arccos_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 6://'arctan'
                _ewise_arctan_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 7://'sign'
                _ewise_sign_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 8://'exp'
                _ewise_exp_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 9://'abs'
                _ewise_abs_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 10://'neg'
                _ewise_neg_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 11://'ceil'
                _ewise_ceil_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 12://'floor'
                _ewise_floor_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 13://'loge'
                _ewise_loge_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 14://'log10'
                _ewise_log10_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            case 15://'log2'
                _ewise_log2_gpu<scalar_t, sscalar_t>(remain_time, c_id, t_n, a, out);
                break;
            default:
                printf("Warning: the input type out of available type");
        }
        c_id += remain_time;
    }
}

///////////////////////////////////////////////////////////////////////
// scalar gpu operator
///////////////////////////////////////////////////////////////////////

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_add_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] + b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_sub_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] - b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_mul_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] * b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_div_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] / b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_pow_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = pow(cash_num_a[c_id + i], b);
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_lt_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] < b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_le_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] <= b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_gt_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] > b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_ge_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] >= b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_ne_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] != b;
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__device__ void _scalar_eq_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, sscalar_t b, tscalar_t* out){
    for(int i = 0; i < remain_size; i += t_n){
        out[c_id + i] = cash_num_a[c_id + i] == b;
    }
}

// __device__ void _scalar_eq_gpu(size_t remain_size, size_t c_id, size_t t_n, scalar_t* cash_num_a, scalar_t* cash_num_b, bool* out){
//     for(int i = 0; i < remain_size; ++i){
//         if (cash_num_b[i] != cash_num_a[i]){
//             out = false;
//             break;
//         }
//     }
//     __threadfence();
//     __syncthreads();
// }

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__global__ void scalar_compute_gpu(scalar_t* a_handle, sscalar_t b, tscalar_t* out, size_t size, int offset_a, size_t type){
    size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t t_n = gridDim.x * blockDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    while(c_id < size){
        size_t remain_size = (size - c_id);
        /////////////////////////////////////////////////////
        // divide the transfer and computation process, to
        // avoid the time cost of data transfer best
        ////////////////////////////////////////////////////
        size_t remain_time = CASH_NUM * t_n;
        if (remain_size < CASH_NUM * t_n){
            remain_time = remain_size;
        }
        switch(type){
            case 0:
                _scalar_add_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 1:
                _scalar_sub_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 2:
                _scalar_mul_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 3:
                _scalar_div_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 4:
                _scalar_pow_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 5:
                _scalar_lt_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 6:
                _scalar_le_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 7:
                _scalar_gt_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 8:
                _scalar_ge_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 9:
                _scalar_ne_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            case 10:
                _scalar_eq_gpu<scalar_t, sscalar_t, tscalar_t>(remain_time, c_id, t_n, a, b, out);
                break;
            default:
                printf("Warning: the input type out of available type");
        }
        c_id += remain_time;
    }
}

///////////////////////////////////////////////////////////////////////
// ewise sum
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_sum(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i];
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise min
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_min(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = a[init_posi];
    internal_output[tid + tn] = a[init_posi];
    for(int i = tid; i < len; i += tn){
        if (internal_output[i % (tn * 2)] > a[init_posi + i]){
            internal_output[i % (tn * 2)] = a[init_posi + i];
        }
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            if(internal_output[tid] > internal_output[tid + i]){
                internal_output[tid] = internal_output[tid + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise max
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_max(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = a[init_posi];
    internal_output[tid + tn] = a[init_posi];
    for(int i = tid; i < len; i += tn){
        if (internal_output[i % (tn * 2)] < a[init_posi + i]){
            internal_output[i % (tn * 2)] = a[init_posi + i];
        }
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            if(internal_output[tid] < internal_output[tid + i]){
                internal_output[tid] = internal_output[tid + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise mean
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_mean(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i];
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0] / len;
    }
}
///////////////////////////////////////////////////////////////////////
// ewise argmax
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_argmax(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ int internal_idx[];
    internal_idx[tid] = 0;
    internal_idx[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        if (a[internal_idx[i % (tn * 2)]] < a[init_posi + i]){
            internal_idx[i % (tn * 2)] = init_posi + i;
        }
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            if(a[internal_idx[tid]] < a[internal_idx[tid + i]]){
                internal_idx[tid] = internal_idx[tid + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_idx[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise argmin
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_argmin(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ int internal_idx[];
    internal_idx[tid] = 0;
    internal_idx[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        if (a[internal_idx[i % (tn * 2)]] > a[init_posi + i]){
            internal_idx[i % (tn * 2)] = init_posi + i;
        }
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            if(a[internal_idx[tid]] > a[internal_idx[tid + i]]){
                internal_idx[tid] = internal_idx[tid + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_idx[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise std
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_std(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i] / len;
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
    __syncthreads();
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += pow((a[init_posi + i] - o[bid]), 2);
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0){
        o[bid] = sqrt(double(internal_output[0] / len));
    }
}
///////////////////////////////////////////////////////////////////////
// ewise var
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_var(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i] / len;
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
    __syncthreads();
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += pow((a[init_posi + i] - o[bid]), 2);
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0){
        o[bid] = internal_output[0] / len;
    }
}
///////////////////////////////////////////////////////////////////////
// ewise cumsum
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_cumsum(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i];
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
}
///////////////////////////////////////////////////////////////////////
// ewise cumprob
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
__global__ void ewise_cumprob(const scalar_t* a_handle, sscalar_t* o, size_t len, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int init_posi = bid * len;
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];
    scalar_t* internal_output = (scalar_t*)(share_tmp);
    internal_output[tid] = 0;
    internal_output[tid + tn] = 0;
    for(int i = tid; i < len; i += tn){
        internal_output[i % (tn * 2)] += a[init_posi + i];
    }
    __syncthreads();
    for(int i = tn; i > 0; i >>= 1){
        if (tid < i){
            internal_output[tid] += internal_output[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        o[bid] = internal_output[0];
    }
}

///////////////////////////////////////////////////////////////////////
// operators with different dims
///////////////////////////////////////////////////////////////////////

template<typename scalar_t>
__device__ void _scalar_compute_dim_1(scalar_t* a, scalar_t b, scalar_t* out, int tid, int len, int t_n, int type){
    switch(type){
        case 0:
            _scalar_add_gpu<scalar_t>(len, tid, t_n, a, b, out);
            break;
        case 1:
            _scalar_sub_gpu<scalar_t>(len, tid, t_n, a, b, out);
            break;
        case 2:
            _scalar_mul_gpu<scalar_t>(len, tid, t_n, a, b, out);
            break;
        case 3:
            _scalar_div_gpu<scalar_t>(len, tid, t_n, a, b, out);
            break;
        default:
            printf("Warning: the input type out of available type");
    }
}

template<typename scalar_t>
__device__ void _scalar_compute_dim_2(scalar_t a, scalar_t* b, scalar_t* out, int tid, int len, int t_n, int type){
    switch(type){
        case 0:
            for(int i = tid; i < len; i += t_n){
                out[i] = a + b[i];
            }
            break;
        case 1:
            for(int i = tid; i < len; i += t_n){
                out[i] = a - b[i];
            }
            break;
        case 2:
            for(int i = tid; i < len; i += t_n){
                out[i] = a * b[i];
            }
            break;
        case 3:
            for(int i = tid; i < len; i += t_n){
                out[i] = a / b[i];
            }
            break;
        default:
            printf("Warning: the input type out of available type");
    }
}

template<typename scalar_t>
__device__ void _ewise_compute_dim(scalar_t* a, scalar_t* b, scalar_t* out, int tid, int len, int t_n, int type){
    switch(type){
        case 0:
            if(len - tid >= 0) _ewise_add_gpu<scalar_t>(len - tid, tid, t_n, a, b, out);
            break;
        case 1:
            if(len - tid >= 0) _ewise_sub_gpu<scalar_t>(len - tid, tid, t_n, a, b, out);
            break;
        case 2:
            if(len - tid >= 0) _ewise_mul_gpu<scalar_t>(len - tid, tid, t_n, a, b, out);
            break;
        case 3:
            if(len - tid >= 0) _ewise_div_gpu<scalar_t>(len - tid, tid, t_n, a, b, out);
            break;
        default:
            printf("Warning: the input type out of available type");
    }
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
__global__ void _operator_dim(scalar_t* a_handle, sscalar_t* b_handle, tscalar_t* out, size_t pre_dim_a, size_t post_dim_a, size_t pre_dim_b, size_t post_dim_b, int offset_a, int offset_b, int type){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int len = post_dim_a;
    scalar_t* a = (scalar_t*)(a_handle + offset_a);
    scalar_t* b = (scalar_t*)(b_handle + offset_b);
    if(post_dim_b == post_dim_a){
        out = (scalar_t*)(out + (bid % pre_dim_a) * len);
        a = (scalar_t*)(a + (bid % pre_dim_a) * len);
        b = (scalar_t*)(b + (bid % pre_dim_b) * post_dim_b);
        // printf("tn:%d\n", tn);
        _ewise_compute_dim<scalar_t>(a, b, out, tid, len, tn, type);
    }
    else if(post_dim_b == 1 && post_dim_a != 1){
        out = (scalar_t*)(out + bid * len);
        a = (scalar_t*)(a + (bid % pre_dim_a) * len);
        scalar_t constant_b = b[bid % pre_dim_b];

        _scalar_compute_dim_1<scalar_t>(a, constant_b, out, tid, len, tn, type);
        // for(int i = tid; i < len; i += tn){
        //     out[i] = a[i] - constant_b;
        // }
    }
    else if(post_dim_a == 1 && post_dim_b != 1){
        len = post_dim_b;
        out = (scalar_t*)(out + bid * len);
        b = (scalar_t*)(b + (bid % pre_dim_b) * len);
        scalar_t constant_a = a[bid % pre_dim_a];

        _scalar_compute_dim_2<scalar_t>(constant_a, b, out, tid, len, tn, type);
    }
}

template<typename scalar_t>
__global__ void transfer_1(scalar_t* out, scalar_t* in, int32_t* idxs, int unit_size, int unit_len, int offset){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tn = blockDim.x * gridDim.x;
    for(int i = tid; i < unit_size * unit_len; i += tn){
        int unit_id = i / unit_len;
        out[i % unit_len + offset + unit_id * unit_len] = in[i % unit_len + idxs[unit_id]];
    }
}

template<typename scalar_t>
__global__ void old_transfer_get(scalar_t* out, scalar_t* in, int32_t* unit_idxs, int unit_size, int unit_len, int offset){
    // printf("here!!!!!!%f, %d\n", in[0], unit_idxs[0]);
    in = (scalar_t*)(in + offset * sizeof(scalar_t));
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tn = blockDim.x * gridDim.x;
    // if(tid == 0){
    //     printf("=================%d\n", unit_size);
    //     for(int i = 0; i < unit_size; ++i){
    //         printf("%f ", in[unit_idxs[i]]);
    //     }
    //     printf("\n=================\n");
    // }
    // __syncthreads();
    for(int i = tid; i < unit_size * unit_len; i += tn){
        int unit_id = i / unit_len;
        out[i] = in[unit_idxs[unit_id] + i % unit_len];
    }
}

template<typename scalar_t>
__global__ void old_transfer_set(scalar_t* out, scalar_t* in, int32_t* unit_idxs, int unit_size, int unit_len, int offset){
    in = (scalar_t*)(in + offset * sizeof(scalar_t));
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tn = blockDim.x * gridDim.x;
    for(int i = tid; i < unit_size * unit_len; i += tn){
        int unit_id = i / unit_len;
        out[unit_idxs[unit_id] + i % unit_len] = in[i];
    }
}

template<typename scalar_t>
__global__ void transfer_set(scalar_t* out, scalar_t* in, int32_t* unit_idxs, int32_t* unit_sizes, int32_t unit_len, int in_offset, int out_offset, int dim){
    in = (scalar_t*)(in + in_offset);
    out = (scalar_t*)(out + out_offset);
    int32_t total_num = unit_len;
    for(int i = 0; i < dim; ++i){
        total_num *= unit_sizes[i];
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tn = blockDim.x * gridDim.x;

    extern __shared__ int32_t remain_nums[];
    int32_t* init_posis = (int32_t*)(remain_nums + dim);
    if (threadIdx.x == 0){
        int32_t remain_num = total_num;
        int32_t init_posi = 0;
        for(int j = 0; j < dim; ++j){
            remain_num /= unit_sizes[j];
            remain_nums[j] = remain_num;
            init_posis[j] = init_posi;
            init_posi += unit_sizes[j];
        }

    }
    __syncthreads();

    for(int i = tid; i < total_num; i += tn){
        int32_t unit_id = 0;
        int32_t posi = i;
        for(int j = 0; j < dim; ++j){
            unit_id += unit_idxs[init_posis[j] + (posi / remain_nums[j])];
            posi %= remain_nums[j];
        }
        unit_id += posi;
        out[unit_id] = in[i];
    }
}

template<typename scalar_t>
__global__ void transfer_get(scalar_t* out, scalar_t* in, int32_t* unit_idxs, int32_t* unit_sizes, int32_t unit_len, int in_offset, int out_offset, int dim){
    in = (scalar_t*)(in + in_offset);
    out = (scalar_t*)(out + out_offset);
    int32_t total_num = unit_len;
    for(int i = 0; i < dim; ++i){
        total_num *= unit_sizes[i];
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tn = blockDim.x * gridDim.x;

    extern __shared__ int32_t remain_nums[];
    int32_t* init_posis = (int32_t*)(remain_nums + dim);
    if (threadIdx.x == 0){
        int32_t remain_num = total_num;
        int32_t init_posi = 0;
        for(int j = 0; j < dim; ++j){
            remain_num /= unit_sizes[j];
            remain_nums[j] = remain_num;
            init_posis[j] = init_posi;
            init_posi += unit_sizes[j];
        }

    }
    __syncthreads();

    for(int i = tid; i < total_num; i += tn){
        int32_t unit_id = 0;
        int32_t posi = i;
        for(int j = 0; j < dim; ++j){
            unit_id += unit_idxs[init_posis[j] + (posi / remain_nums[j])];
            posi %= remain_nums[j];
        }
        unit_id += posi;
        out[i] = in[unit_id];
    }
}


///////////////////////////////////////////////////////////////////////
// bind the computation with corresponding stream
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t, typename fscalar_t>
size_t ewise_async_3op(const Array<scalar_t>& a, const Array<sscalar_t>& b, const Array<tscalar_t>& c, Array<fscalar_t>& out){
    size_t block_num = (out.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if(a.stream_id == -1 && b.stream_id == -1 && c.stream_id == -1){
        if(out.stream_id == -1) out.stream_id = rand() % STREAM_NUM_NDARRAY;
    }
    else if(out.stream_id != -1){
        cudaStreamWaitEvent(streams[out.device_id][out.stream_id], a.event_sign, 0);
        cudaStreamWaitEvent(streams[out.device_id][out.stream_id], b.event_sign, 0);
        cudaStreamWaitEvent(streams[out.device_id][out.stream_id], c.event_sign, 0);
    }
    else{
        if(a.stream_id != -1){
            out.stream_id = a.stream_id;
            cudaStreamWaitEvent(streams[out.device_id][out.stream_id], b.event_sign, 0);
        }
        else if(b.stream_id != -1){
            out.stream_id = b.stream_id;
            cudaStreamWaitEvent(streams[out.device_id][out.stream_id], b.event_sign, 0);
            cudaStreamWaitEvent(streams[out.device_id][out.stream_id], c.event_sign, 0);
        }
        else{
            out.stream_id = c.stream_id;
        }
    }
    return block_num;
}

template<typename scalar_t, typename sscalar_t>
void ewise_assign(Array<scalar_t>& a, sscalar_t val, int post_dim, int offset){
    size_t block_num = (a.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if(a.stream_id == -1) a.stream_id = rand() % STREAM_NUM_NDARRAY;

    scalar_t* a_gpu = a.ptr;
    ewise_assign_global<scalar_t, sscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[a.device_id][a.stream_id]>>>(a_gpu, val, post_dim, offset);

    cudaEventRecord(a.event_sign, streams[a.device_id][a.stream_id]);
}

template<typename scalar_t, typename sscalar_t>
void ewise_uniform(Array<scalar_t>& a, sscalar_t val, int post_dim, int offset){
    size_t block_num = (a.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if(a.stream_id == -1) a.stream_id = rand() % STREAM_NUM_NDARRAY;

    scalar_t* a_gpu = a.ptr;
    ewise_assign_global<scalar_t, sscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[a.device_id][a.stream_id]>>>(a_gpu, val, post_dim, offset);

    cudaEventRecord(a.event_sign, streams[a.device_id][a.stream_id]);
}

///////////////////////////////////////////////////////////////////////
// bind the computation with corresponding stream
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t>
size_t scalar_async(const Array<scalar_t>& a, sscalar_t b, Array<tscalar_t>& out){
    size_t block_num = (out.size + INIT_THREAD_NUM * CASH_NUM - 1) / (INIT_THREAD_NUM * CASH_NUM);
    if (a.stream_id == -1){
        if(out.stream_id == -1) out.stream_id = rand() % STREAM_NUM_NDARRAY;
        else cudaStreamWaitEvent(streams[out.device_id][out.stream_id], a.event_sign, 0);
    }
    else{
        out.stream_id = a.stream_id;
    }
    return block_num;
}

///////////////////////////////////////////////////////////////////////
// transfer from cpu to gpu
///////////////////////////////////////////////////////////////////////
template<typename scalar_t>
void to_gpu(const Array<scalar_t>& a, scalar_t* a_gpu, cudaStream_t* stream){
     
    // cudaMallocAsync((void**)&a_gpu, a.size * ELEM_SIZE, *stream);
    // cudaMemcpyAsync(a_gpu, a_ptr, a.size * ELEM_SIZE, cudaMemcpyHostToDevice, *stream);
}

///////////////////////////////////////////////////////////////////////
// ewise gpu computation
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t, typename fscalar_t>
void ewise_compute_3op(const Array<scalar_t>& a_handle, const Array<sscalar_t>& b_handle, const Array<tscalar_t>& c_handle, Array<fscalar_t>& out, int32_t offset_a, int32_t offset_b, int32_t offset_c, size_t op_type){
    Array<scalar_t> a(0, out.device_id);
    Array<sscalar_t> b(0, out.device_id);
    Array<tscalar_t> c(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    sscalar_t *b_ptr = cpy_gpus(b_handle, out, b);
    tscalar_t *c_ptr = cpy_gpus(c_handle, out, c);
    size_t block_num = ewise_async_3op(a, b, c, out);

    scalar_t* a_gpu;
    sscalar_t *b_gpu;
    tscalar_t *c_gpu;
    fscalar_t *out_gpu;
    a_gpu = a_ptr;
    b_gpu = b_ptr;
    c_gpu = c.ptr;
    out_gpu = out.ptr;

    ewise_compute_gpu_3op<scalar_t, sscalar_t, tscalar_t, fscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[out.device_id][out.stream_id]>>>(a_gpu, b_gpu, c_gpu, out_gpu, out.size, offset_a, offset_b, offset_c, op_type);

    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

///////////////////////////////////////////////////////////////////////
// ewise gpu computation
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void ewise_compute_2op(const Array<scalar_t>& a_handle, const Array<sscalar_t>& b_handle, Array<tscalar_t>& out, int32_t offset_a, int32_t offset_b, size_t op_type){
    Array<scalar_t> a(0, out.device_id);
    Array<sscalar_t> b(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    sscalar_t *b_ptr = cpy_gpus(b_handle, out, b);
    size_t block_num = ewise_async_2op(a, b, out);

    scalar_t* a_gpu;
    sscalar_t *b_gpu;
    tscalar_t *out_gpu;
    /* [ ] TODO: Since the current CUDA driver version is less than 11.2, we can not use cudaMallocAsync and cudaFreeAsync,
     then we prealloc all tensor in temporary. it should be changed latter*/
    // to_gpu(a, a_gpu, out.stream);
    // to_gpu(b, b_gpu, out.stream);
    // to_gpu(out, out_gpu, out.stream);
    a_gpu = a_ptr;
    b_gpu = b_ptr;
    out_gpu = out.ptr;

    ewise_compute_gpu_2op<scalar_t, sscalar_t, tscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[out.device_id][out.stream_id]>>>(a_gpu, b_gpu, out_gpu, out.size, offset_a, offset_b, op_type);

    /* [ ] TODO: The following row should be restored after using async alloc.*/
    // cudaMemcpyAsync(out.ptr, out_gpu, out.size * ELEM_SIZE, cudaMemcpyDeviceToHost, *out.stream);
    /* insert a event to current stream, to get the Array<scalar_t> when the kernel finish */
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    /* [ ] TODO: The following three rows should be restored after using async alloc.*/
    // cudaFreeAsync(a_gpu, *out.stream);
    // cudaFreeAsync(b_gpu, *out.stream);
    // cudaFreeAsync(out_gpu, *out.stream);
}

template<typename scalar_t, typename sscalar_t>
void ewise_compute_1op(const Array<scalar_t>& a_handle, Array<sscalar_t>& out, int32_t offset_a, size_t op_type){

    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    size_t block_num = ewise_async_1op(a, out);
    
    scalar_t* a_gpu;
    sscalar_t* out_gpu;
    /* [ ] TODO: Since the current CUDA driver version is less than 11.2, we can not use cudaMallocAsync and cudaFreeAsync,
     then we prealloc all tensor in temporary. it should be changed latter*/
    // to_gpu(a, a_gpu, out.stream);
    // to_gpu(out, out_gpu, out.stream);
    
    a_gpu = a_ptr;
    out_gpu = out.ptr;
    ewise_compute_gpu_1op<scalar_t, sscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[out.device_id][out.stream_id]>>>(a_gpu, out_gpu, out.size, offset_a, op_type);
    /* [ ] TODO: The following row should be restored after using async alloc.*/
    // cudaMemcpyAsync(out.ptr, out_gpu, out.size * ELEM_SIZE, cudaMemcpyDeviceToHost, *out.stream);
    /* insert a event to current stream, to get the Array<scalar_t> when the kernel finish */
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    /* [ ] TODO: The following three rows should be canceled after using async alloc.*/
    // cudaFreeAsync(a_gpu, *out.stream);
    // cudaFreeAsync(b_gpu, *out.stream);
    // cudaFreeAsync(out_gpu, *out.stream);
}

///////////////////////////////////////////////////////////////////////
// scalar gpu operator
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void scalar_compute_2op(const Array<scalar_t>& a, sscalar_t b, Array<tscalar_t>& out, int offset_a, size_t op_type){
    size_t block_num = scalar_async(a, b, out);

    scalar_t* a_gpu;
    tscalar_t* out_gpu;
    /* [ ] TODO: Since the current CUDA driver version is less than 11.2, we can not use cudaMallocAsync and cudaFreeAsync,
     then we prealloc all tensor in temporary. it should be changed latter*/
    // to_gpu(a, a_gpu, out.stream);
    // to_gpu(out, out_gpu, out.stream);
    
    a_gpu = a.ptr;
    out_gpu = out.ptr;
    scalar_compute_gpu<scalar_t, sscalar_t, tscalar_t><<<block_num, INIT_THREAD_NUM, 0, streams[out.device_id][out.stream_id]>>>(a_gpu, b, out_gpu, out.size, offset_a, op_type);
    
    /* [ ] TODO: The following row should be restored after using async alloc.*/
    // cudaMemcpyAsync(out.ptr, out_gpu, out.size * ELEM_SIZE, cudaMemcpyDeviceToHost, *out.stream);
    /* insert a event to current stream, to get the Array<scalar_t> when the kernel finish */
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    /* [ ] TODO: The following two rows should be canceled after using async alloc.*/
    // cudaFreeAsync(a_gpu, *out.stream);
    // cudaFreeAsync(out_gpu, *out.stream);
}

// __global__ void ewise_judge_gpu(scalar_t* a, scalar_t* b, bool* out, size_t size, size_t type){
//     size_t tid = threadIdx.x;
//     size_t bid = blockIdx.x;
//     size_t c_id = threadIdx.x + blockIdx.x * blockDim.x;
//     const size_t t_n = gridDim.x * blockDim.x;

//     scalar_t cash_num_a[CASH_NUM], cash_num_b[CASH_NUM];

//     while(c_id < size){
//         size_t remain_size = (size - c_id);
//         /////////////////////////////////////////////////////
//         // divide the transfer and computation process, to
//         // avoid the time cost of data transfer best
//         ////////////////////////////////////////////////////
//         if (remain_size > 0){
//             remain_size = remain_size / t_n + 1;
//         }
//         else{
//             remain_size = 0;
//         }
//         for(int i = 0; i < remain_size; ++i){
//             cash_num_a[i] = a[c_id];
//             cash_num_b[i] = b[c_id];
//         }
//         __syncthreads();
//         switch(type){
//             case 0:
//                 _ewise_eq_gpu(remain_size, c_id, t_n, cash_num_b, cash_num_b, out);
//                 if (out == false){
//                     return;
//                 }
//                 break;
//             default:
//                 printf("Warning: the input type out of available type");
//         }
//         __syncthreads();
//         c_id += remain_size * t_n;
//     }
// }

// void ewise_judge_2op(const Array<scalar_t>& a, scalar_t b, bool& out, size_t op_type){
//     size_t block_num = scalar_async(a, b, out);
//
//     scalar_t* a_gpu, *out_gpu;
//     to_gpu(a, a_gpu, out.stream);
//     to_gpu(out, out_gpu, out.stream);
//
//     scalar_compute_gpu<<<block_num, INIT_THREAD_NUM, 0, *out.stream>>>(a_gpu, b, out_gpu, out.size, op_type);
//
//     cudaMemcpyAsync(out.ptr, out_gpu, cudaMemcpyDeviceToHost, out.stream);
//     /* insert a event to current stream, to get the Array<scalar_t> when the kernel finish */
//     cudaEventCreate(&out.event_sign);
//     cudaEventRecord(out.event_sign, out.stream);
//
//     cudaFreeAsync(a_gpu, out.stream);
//     cudaFreeAsync(out_gpu, out.stream);
// }

///////////////////////////////////////////////////////////////////////
// cublas sum gpu operator
///////////////////////////////////////////////////////////////////////
template<typename scalar_t, typename sscalar_t>
void oper_dim_1(const Array<scalar_t>& a_handle, Array<sscalar_t>& out, int pre_dim, int post_dim, int offset, size_t op_type){
    
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    const size_t ELEM_SIZE = sizeof(scalar_t);
    size_t block_num = ewise_async_1op(a, out);
    int thread_num = 1024;
    if (post_dim < thread_num){
        thread_num = post_dim + 32 - post_dim % 32;
    }
    switch(op_type){
        case 0:
            ewise_sum<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 1:
            ewise_min<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 2:
            ewise_max<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 3:
            ewise_mean<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 4:
            ewise_argmax<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * sizeof(int) * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 5:
            ewise_argmin<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * sizeof(int) * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 6:
            ewise_std<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        case 7:
            ewise_var<scalar_t, sscalar_t><<<pre_dim, thread_num, thread_num * ELEM_SIZE * 2, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, post_dim, offset);
            break;
        default:
            std::cerr << "The operator " << op_type << " is not implemented yet.\n";
    }
    // // double* n_tmp = (double*)malloc(sizeof(double) * pre_dim);
    // cublasStatus_t ret;
    // // cudaMalloc(&n_tmp, sizeof(double) * pre_dim);
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSetStream(handle, streams[out.device_id][out.stream_id]);
    // printf("!!!!!");
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    // for(int i = 0; i < pre_dim; ++i){
    //     ret = cublasDasum(handle, post_dim, a_ptr + i * post_dim, 1, out.ptr + i);
    //     cudaDeviceSynchronize();
    // }
    // printf("!!!!!");
    // cublasDestroy(handle);
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void operator_dim(const Array<scalar_t>& a_handle, const Array<sscalar_t>& b_handle, Array<tscalar_t>& out, int pre_dim_a, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b, int type){
    
    Array<scalar_t> a(0, out.device_id);
    Array<sscalar_t> b(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    sscalar_t *b_ptr = cpy_gpus(b_handle, out, b);
    // scalar_t* a_ptr = a.ptr, *b_ptr = b.ptr;
    
    size_t block_num = ewise_async_2op(a, b, out);
    int thread_num = 1024;
    int post_dim = max(post_dim_a, post_dim_b);
    if (post_dim < thread_num){
        thread_num = post_dim + 32 - post_dim % 32;
    }
    // assert(post_dim_a == post_dim_b || post_dim_a == 1 || post_dim_b == 1);
    // assert(pre_dim_a % pre_dim_b == 0);
    // printf("%d, %d, %d\n", int(a.size), int(b.size), int(out.size));
    // assert(pre_dim_a * post_dim_a == a.size && pre_dim_b * post_dim_b == b.size);
    // assert("the post dim of two inputs should keep same: %d, %d", post_dim_a, post_dim_b);
    _operator_dim<scalar_t, sscalar_t, tscalar_t><<<pre_dim_a, thread_num, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, b_ptr, out.ptr, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, type);
    
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t>
__global__ void mem_addr_pointer(scalar_t* a_handle, scalar_t** addrs, int pre_dim, int size, int offset){
    int tid = threadIdx.x + blockIdx.x * blockDim.x, t_n = blockDim.x * gridDim.x;
    scalar_t* a = (scalar_t*)(a_handle + offset);
    while(tid < pre_dim){
        addrs[tid] = (scalar_t*)(a + tid * size);
        
        tid += t_n;
    }
}

template<typename scalar_t>
__global__ void diagonal_sum(scalar_t* a_handle, scalar_t* out, int pre_dim, int n, int offset){
    scalar_t* a = (scalar_t*)(a_handle + offset);
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char share_tmp[];//n * pre_dim
    scalar_t* tmp = (scalar_t*)(share_tmp);
    int size = n * n, shared_size = n;
    int bid = blockIdx.x;
    int tid = threadIdx.x, t_n = blockDim.x, b_n = gridDim.x;
    for(int k = bid; k < pre_dim; k += b_n){
        while(tid < shared_size){
            tmp[tid] = a[k * size + tid * n + tid % n];
            tid += t_n;
        }
        __syncthreads();
        for(int i = n / 2; i > 0; i >>= 1){
            tid = threadIdx.x;
            while (tid < i){
                tmp[tid] += tmp[tid + i];
                tid += t_n;
            }
            if(i % 2 == 1 && threadIdx.x == 0){
                tmp[0] += tmp[i - 1];
            }
            __syncthreads();
        }
        out[k] = tmp[0];
    }
}

template<typename scalar_t>
void matrix_inv(Array<scalar_t>& a_handle, Array<scalar_t>& out, int pre_dim, int col_num_a, int post_dim_a, int offset_a, Array<scalar_t>& infos){
    
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);

    ewise_async_1op(a, out);

    void * src_ptr;
    mem_pool_alloc(pre_dim * sizeof(scalar_t*), src_ptr, out.device_id, out.stream_id);
    scalar_t ** srcDptr = reinterpret_cast<scalar_t**>(src_ptr);
    
    void * out_ptr;
    mem_pool_alloc(pre_dim * sizeof(scalar_t*), out_ptr, out.device_id, out.stream_id);
    scalar_t ** outDptr = reinterpret_cast<scalar_t**>(out_ptr);
    
    void* pivot_scalar;
    mem_pool_alloc(sizeof(int) * col_num_a * pre_dim, pivot_scalar, out.device_id, out.stream_id);
    int *pivot = reinterpret_cast<int*>(pivot_scalar);

    mem_addr_pointer<scalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, srcDptr, pre_dim, post_dim_a, offset_a);
    mem_addr_pointer<scalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, outDptr, pre_dim, post_dim_a, 0);
    cublasSetStream_v2(cublas_handle, streams[out.device_id][out.stream_id]);
    if (typeid(scalar_t) == typeid(float) || typeid(scalar_t) == typeid(int32_t)){
        cublasSgetrfBatched(
            cublas_handle, col_num_a, (float**)srcDptr, col_num_a, pivot, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
        cublasSgetriBatched(
            cublas_handle, col_num_a, (float**)srcDptr, col_num_a, pivot, (float**)outDptr, col_num_a, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
    }
    else if(typeid(scalar_t) == typeid(double) || typeid(scalar_t) == typeid(int64_t)){
        cublasDgetrfBatched(
            cublas_handle, col_num_a, (double**)srcDptr, col_num_a, pivot, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
        cublasDgetriBatched(
            cublas_handle, col_num_a, (double**)srcDptr, col_num_a, pivot, (double**)outDptr, col_num_a, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
    }
    else{
        std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    }

    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    cuda_mem_pool.free(pre_dim * sizeof(scalar_t*), (void*)src_ptr, out.device_id, out.event_sign);
    cuda_mem_pool.free(pre_dim * sizeof(scalar_t*), (void*)out_ptr, out.device_id, out.event_sign);
    cuda_mem_pool.free(sizeof(int) * col_num_a * pre_dim, (void*)pivot_scalar, out.device_id, out.event_sign);
    
}

template<typename scalar_t>
void matrix_diagonal_sum(const Array<scalar_t>& a_handle, Array<scalar_t>& out, int pre_dim, int n, int offset_a){
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);
    
    int thread_num = 1024;
    if (n < thread_num){
        thread_num = n + 32 - n % 32;
    }
    int block_num = 32;
    if (pre_dim < block_num){
        block_num = pre_dim;
    }
    diagonal_sum<scalar_t><<<block_num, thread_num, block_num * n, streams[out.device_id][out.stream_id]>>>(a_ptr, out.ptr, pre_dim, n, offset_a);
    
    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
}

template<typename scalar_t>
void matrix_det(Array<scalar_t>& a_handle, Array<scalar_t>& out, int pre_dim, int col_num_a, int post_dim_a, int offset_a, Array<scalar_t>& infos){
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);

    void * src_ptr;
    mem_pool_alloc(pre_dim * sizeof(scalar_t*), src_ptr, out.device_id, out.stream_id);
    scalar_t ** srcDptr = reinterpret_cast<scalar_t**>(src_ptr);
    
    mem_addr_pointer<scalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, srcDptr, pre_dim, post_dim_a, offset_a);

    void* pivot_scalar;
    mem_pool_alloc(sizeof(int) * col_num_a * pre_dim, pivot_scalar, out.device_id, out.stream_id);
    int *pivot = reinterpret_cast<int*>(pivot_scalar);

    cublasSetStream_v2(cublas_handle, streams[out.device_id][out.stream_id]);
    if (typeid(scalar_t) == typeid(float) || typeid(scalar_t) == typeid(int32_t)){
        cublasSgetrfBatched(
            cublas_handle, col_num_a, (float**)srcDptr, col_num_a, pivot, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
    }
    else if(typeid(scalar_t) == typeid(double) || typeid(scalar_t) == typeid(int64_t)){
        cublasDgetrfBatched(
            cublas_handle, col_num_a, (double**)srcDptr, col_num_a, pivot, reinterpret_cast<int*>(infos.ptr), pre_dim
        );
    }
    else{
        std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    }
    matrix_diagonal_sum<scalar_t>(a, out, pre_dim, col_num_a, offset_a);

    cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    cuda_mem_pool.free(sizeof(int) * col_num_a * pre_dim, (void*)pivot_scalar, out.device_id, out.event_sign);
    cuda_mem_pool.free(pre_dim * sizeof(scalar_t*), (void*)src_ptr, out.device_id, out.event_sign);

}

template<typename scalar_t>
void matrix_transpose(const Array<scalar_t>& a_handle, Array<scalar_t>& out, int pre_dim, int col_num_a, int post_dim_a, int offset_a){
    Array<scalar_t> a(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    ewise_async_1op(a, out);

    int row_num_a = post_dim_a / col_num_a;
    cublasSetStream_v2(cublas_handle, streams[out.device_id][out.stream_id]);
    cublasStatus_t status;
     
    for(int i = 0; i < pre_dim; ++i){
        if (typeid(scalar_t) == typeid(float) || typeid(scalar_t) == typeid(int32_t)){
            float alpha = 1.0, beta = 0;
            status = cublasSgeam(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                row_num_a, col_num_a,
                &alpha,
                (float*)(a_ptr + i * post_dim_a + offset_a), col_num_a,
                &beta,
                (float*)(a_ptr + i * post_dim_a), col_num_a,
                (float*)(out.ptr + i * post_dim_a),
                row_num_a
            );
        }
        else if (typeid(scalar_t) == typeid(double) || typeid(scalar_t) == typeid(int64_t)){
            double alpha = 1.0, beta = 0;
            status = cublasDgeam(
                cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                row_num_a, col_num_a,
                &alpha,
                (double*)(a_ptr + i * post_dim_a + offset_a), col_num_a,
                &beta,
                (double*)(a_ptr + i * post_dim_a), col_num_a,
                (double*)(out.ptr + i * post_dim_a),
                row_num_a
            );
        }
        else{     
            std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
        }
    }

    if (status != CUBLAS_STATUS_SUCCESS){
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        std::cout << "code: " << status << std::endl;
        exit(1);
    }
    
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void matrix_opers(const Array<scalar_t>& a_handle, const Array<sscalar_t>& b_handle, Array<tscalar_t>& out,
     int pre_dim, int col_num_a, int col_num_b, int post_dim_a, int post_dim_b, 
     int offset_a, int offset_b){
        
    Array<scalar_t> a(0, out.device_id);
    Array<sscalar_t> b(0, out.device_id);
    scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    sscalar_t *b_ptr = cpy_gpus(b_handle, out, b);
    ewise_async_2op(a, b, out);

    int row_num_a = post_dim_a / col_num_a, row_num_b = post_dim_b / col_num_b;
    
    cublasSetStream_v2(cublas_handle, streams[out.device_id][out.stream_id]);
    cublasStatus_t status;
    if(pre_dim > 1){
        void* a_pts, *b_pts, *c_pts;
        mem_pool_alloc(sizeof(scalar_t*) * pre_dim, a_pts, out.device_id, out.stream_id);
        mem_pool_alloc(sizeof(sscalar_t*) * pre_dim, b_pts, out.device_id, out.stream_id);
        mem_pool_alloc(sizeof(tscalar_t*) * pre_dim, c_pts, out.device_id, out.stream_id);
        scalar_t** a_addrs = reinterpret_cast<scalar_t**>(a_pts);
        sscalar_t **b_addrs = reinterpret_cast<sscalar_t**>(b_pts);
        tscalar_t **c_addrs = reinterpret_cast<tscalar_t**>(c_pts);
        mem_addr_pointer<scalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(a_ptr, a_addrs, pre_dim, post_dim_a, offset_a);
        mem_addr_pointer<sscalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(b_ptr, b_addrs, pre_dim, post_dim_b, offset_b);
        mem_addr_pointer<tscalar_t><<<1, int(pre_dim / 32) * 32 < 1024 ? int(pre_dim / 32) * 32 : 1024, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, c_addrs, pre_dim, row_num_a * col_num_b, 0);
        if (typeid(scalar_t) == typeid(double) || typeid(scalar_t) == typeid(int64_t)){
            double alpha = 1.0, beta = 0;
            status = cublasDgemmBatched(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                col_num_b, row_num_a, col_num_a,
                &alpha,  
                (double**)b_addrs, col_num_b,
                (double**)a_addrs, col_num_a,
                &beta,
                (double**)c_addrs,
                col_num_b,
                pre_dim
            );
        }
        else{
            float alpha = 1.0, beta = 0;
            status = cublasSgemmBatched(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                col_num_b, row_num_a, col_num_a,
                &alpha,  
                (float**)b_addrs, col_num_b,
                (float**)a_addrs, col_num_a,
                &beta,
                (float**)c_addrs,
                col_num_b,
                pre_dim
            );
        }
        
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
        cuda_mem_pool.free(sizeof(scalar_t*) * pre_dim, a_pts, out.device_id, out.event_sign);
        cuda_mem_pool.free(sizeof(scalar_t*) * pre_dim, b_pts, out.device_id, out.event_sign);
        cuda_mem_pool.free(sizeof(scalar_t*) * pre_dim, c_pts, out.device_id, out.event_sign);
    }
    else{
        if (typeid(scalar_t) == typeid(double) || typeid(scalar_t) == typeid(int64_t)){
            double alpha = 1.0, beta = 0;
            status = cublasDgemm_v2(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                col_num_b, row_num_a, col_num_a,
                &alpha,  
                (double*)b_ptr, col_num_b,
                (double*)a_ptr, col_num_a,
                &beta,
                (double*)out.ptr,
                col_num_b
            );
        }
        else{
            float alpha = 1.0, beta = 0;
            status = cublasSgemm_v2(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                col_num_b, row_num_a, col_num_a,
                &alpha,  
                (float*)b_ptr, col_num_b,
                (float*)a_ptr, col_num_a,
                &beta,
                (float*)out.ptr,
                col_num_b
            );
        }
    }

    if (status != CUBLAS_STATUS_SUCCESS){
        printf("ERROR: %s:%d,", __FILE__, __LINE__);
        std::cout << "code: " << status << std::endl;
        exit(1);
    }
}
}

namespace pygp_img{
    
    struct MatrixShape{
        int32_t width;
        int32_t height;
        MatrixShape(int32_t width, int32_t height){
            this->width = width;
            this->height = height;
        }
    };

    // std::vector<NppiMaskSize> mask_size = {
    //     NPP_MASK_SIZE_1_X_3,
    //     NPP_MASK_SIZE_1_X_5,
    //     NPP_MASK_SIZE_3_X_1,
    //     NPP_MASK_SIZE_5_X_1,
    //     NPP_MASK_SIZE_5_X_5,
    //     NPP_MASK_SIZE_7_X_7,
    //     NPP_MASK_SIZE_9_X_9,
    //     NPP_MASK_SIZE_11_X_11,
    //     NPP_MASK_SIZE_13_X_13,
    //     NPP_MASK_SIZE_15_X_15
    // };
    // [ ] TODO: The mask and roi_npp may meet mistake in border
    #define CONV_INPUT_SHARED 512
    template<typename scalar_t, typename sscalar_t>
    __global__ void convolution(scalar_t* a_handle, sscalar_t* kernel_handle, scalar_t* out, MatrixShape kernel_size, MatrixShape a_size, MatrixShape padding, int32_t dilation, int32_t stride, int32_t constant, int32_t offset_a, int32_t offset_k, int32_t channel_num){
        scalar_t* a = (scalar_t*)(a_handle + offset_a);
        sscalar_t* kernel = (sscalar_t*)(kernel_handle + offset_k);
        int a_width = (a_size.width + 2 * padding.width);
        int size_a_w = (a_width - kernel_size.width) / stride + 1;
        int tid = threadIdx.x, t_n = blockDim.x, a_pointer = 0, kernel_len = kernel_size.width * kernel_size.height, a_len = a_width * a_size.height, out_size=size_a_w * (a_size.height - kernel_size.height + 1);
        out = out + blockIdx.x * size_a_w * (a_size.height - kernel_size.height + 1);
        int k_width = kernel_size.width * (dilation + 1) - 1;
        // [ ] TODO: Not consider the condition that k_width larger than 1024.  
        int a_len_shared = (CONV_INPUT_SHARED - CONV_INPUT_SHARED % k_width) < a_len ? (CONV_INPUT_SHARED - CONV_INPUT_SHARED % k_width):a_len;
        
        extern __shared__ double shared_mm[];
        sscalar_t* kernel_shared = (sscalar_t*)shared_mm;
        scalar_t* a_shared = (scalar_t*)(kernel_shared + (int((kernel_len * sizeof(sscalar_t)) / sizeof(double)) + 1) * sizeof(double));
        float* reduce_shared = (float*)(a_shared + ((int)(a_len_shared * sizeof(scalar_t) / sizeof(double)) + 1) * sizeof(double));//blockDim.x / kernel_size.width * sizeof(float)

        for(int i = tid; i < kernel_len; i += t_n){
            kernel_shared[i] = kernel[i];
        }
        for(int i = tid; i < size_a_w * (a_size.height - kernel_size.height + 1); i += t_n){
            out[i] = 0;
        }
        int reduced_mm = blockDim.x / kernel_size.width + 1;
        for(int i = tid; i < reduced_mm; i += t_n){
            reduce_shared[i] = 0;
        }
        while(a_len_shared != 0){
            for(int i = tid, j = tid + a_pointer; i < a_len_shared; i += t_n, j += t_n){
                if(i % a_width < padding.width || i % a_width >= a_size.width + padding.width){
                    a_shared[i] = constant;
                }
                else{
                    a_shared[i] = a[j];
                }
            }
            __syncthreads();
            // ((a_pointer - a_width) / a_width) * size_a_w;
            int l, a_posi;
            for(int i = tid, min_id = 0; ; i += t_n, min_id += t_n){
                l = (i * stride);
                a_posi = int(l / size_a_w) * a_width + l;
                if(a_posi + kernel_size.width >= a_len_shared){
                    break;
                }
                for(int j = 0; j < kernel_size.height; ++j){
                    int out_posi = ((a_pointer - a_width * j) / a_width) * size_a_w + l;
                    if(((a_pointer + a_width * (kernel_size.height - j)) / a_width) * size_a_w < out_size - l && out_posi >= 0){
                        for(int z = 0; z < kernel_size.width; ++z){
                            a_posi += z;
                            out[out_posi] += (float)(a_shared[a_posi] * kernel_shared[z % kernel_size.width + kernel_size.width * j]);
                        }
                        // __syncthreads();
                    }
                }
                
            }

            a_pointer += a_len_shared;//a_len_shared < (a_len - a_len_shared) ? a_len_shared:(a_len - a_len_shared); 
            if (a_len_shared > a_len - a_pointer){
                a_len_shared = a_len - a_pointer;
            }
        }
    }

    
    // template<typename scalar_t, typename sscalar_t>
    // __global__ void max_pool(scalar_t* a, sscalar_t** kernel, scalar_t* out, MatrixShape kernel_size, MatrixShape a_size, int32_t dilation){
    //     int thread_ona_posi = threadIdx.x % kernel_size + threadIdx.x * kernel_size;
    // }

    // template<typename scalar_t>
    // void gaussian_filter_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
        
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterGauss_8u_C1R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                
    //             nppiFilterGauss_16u_C1R(
    //                 (Npp16u*)(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    // template<typename scalar_t>
    // void gaussian_filter_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterGauss_8u_C3R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                
    //             nppiFilterGauss_16u_C3R(
    //                 (Npp16u*)(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    // template<typename scalar_t>
    // void gaussian_filter_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterGauss_8u_C4R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                
    //             nppiFilterGauss_16u_C4R(
    //                 (Npp16u*)(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }

    // template<typename scalar_t>
    // void laplacian_filter_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
                
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterLaplace_8u_C1R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(int16_t)){
                
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                
    //             nppiFilterLaplace_16s_C1R(
    //                 reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16s*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    // template<typename scalar_t>
    // void laplacian_filter_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
                
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterLaplace_8u_C3R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(int16_t)){
                
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                
    //             nppiFilterLaplace_16s_C3R(
    //                 reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16s*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    // template<typename scalar_t>
    // void laplacian_filter_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, int pre_dim, int post_dim){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void * cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterLaplace_8u_C4R(
    //                 reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(int16_t)){
    //             mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //             nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //             nppiFilterLaplace_16s_C4R(
    //                 reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                 nstep_a * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16s*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_size[mask]
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    
    // template<typename scalar_t>
    // void sobel_filter_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int pre_dim, int post_dim, bool horiz=true){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a), *out_ptr;
    //     void* cSrc_tmp = nullptr;
    //     // cudaDeviceSynchronize();
    //     // cudaError_t err = cudaGetLastError();
    //     // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    //     // printf("0000\n");
    //     ewise_async_1op(a, out);
    //     // cudaDeviceSynchronize();
    //     // err = cudaGetLastError();
    //     // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    //     // printf("-1-1-1-1\n");
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     // cudaDeviceSynchronize();
    //     // err = cudaGetLastError();
    //     // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    //     // printf("-2-2-2-2\n");
        
    //     // assert(pre_dim * post_dim <= a_handle.size && pre_dim * post_dim <= out.size);
    //     for(int z = 0; z < pre_dim; ++z){
    //         // cudaDeviceSynchronize();
    //         // err = cudaGetLastError();
    //         // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    //         // printf("1111, %d, %d, %d, %d, %d, %d, %d, %d\n", z, post_dim, pre_dim, nstep_a, a_handle.size, out.size, roi_npp.width, roi_npp.height);
    //         out_ptr = out.ptr + z * post_dim;
    //         if (horiz){
    //             // state_check("0");
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_8u_C1R(
    //                     reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //                 // state_check("2");
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
                    
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_16s_C1R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //             // state_check("1");
    //         }
    //         else{
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_8u_C1R(
    //                     reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_16s_C1R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //         }
            
    //         // state_check("2");
    //         if (z % OVERLAP_IMG_TIME == 0 && z != 0){
    //             cudaDeviceSynchronize();
    //         }
    //         // state_check("3");
    //         // cudaDeviceSynchronize();
    //         // cudaError_t err = cudaGetLastError();
    //         // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    //         // printf("2222\n");
    //     }
    // }
    // template<typename scalar_t>
    // void sobel_filter_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int pre_dim, int post_dim, bool horiz=true){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void* cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if (horiz){
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_8u_C3R(
    //                     reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_16s_C3R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //         }
    //         else{
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_8u_C3R(
    //                     (Npp8u*)(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_16s_C3R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }
    // template<typename scalar_t>
    // void sobel_filter_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int pre_dim, int post_dim, bool horiz=true){
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     void* cSrc_tmp = nullptr;
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         if (horiz){
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_8u_C4R(
    //                     reinterpret_cast<Npp8u*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelHoriz_16s_C4R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //         }
    //         else{
    //             if(typeid(scalar_t) == typeid(uint8_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_8u_C4R(
    //                     (Npp8u*)(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp8u*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else if(typeid(scalar_t) == typeid(int16_t)){
    //                 mem_pool_alloc((roi_npp.width + 2) * (roi_npp.height + 2) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //                 Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //                 NppiSize c_roi_npp = {roi_npp.width + 2, roi_npp.height + 2};
    //                 nppiCopyConstBorder_16u_C1R(reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + 2) * sizeof(scalar_t), c_roi_npp, 1, 1, 0);
                    
    //                 nppiFilterSobelVert_16s_C4R(
    //                     reinterpret_cast<Npp16s*>(cSrc + nstep_a + 2 + 1),
    //                     nstep_a * sizeof(scalar_t),
    //                     reinterpret_cast<Npp16s*>(out_ptr),
    //                     nstep_out * sizeof(scalar_t),
    //                     roi_npp
    //                 );
    //             }
    //             else{
    //                 std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //             }
    //         }
    //         cudaDeviceSynchronize();
    //     }
    // }

    // template<typename scalar_t>
    // void median_filter_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
                
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     void* cSrc_tmp = nullptr;
        
    //     Npp32u buffer_size;
    //     for(int z = 0; z < pre_dim; ++z){
    //         // printf("1111, %d, %d, %d, %d, %d, %d, %d, %d\n", z, post_dim, pre_dim, nstep_a, a_handle.size, out.size, roi_npp.width, roi_npp.height);
            
    //         void* buffer_tmp = nullptr;
    //         out_ptr = out.ptr + z * post_dim;
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         if(typeid(scalar_t) == typeid(uint8_t)){
                
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             // nppiCopyWrapBorder_8u_C1R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x, 0);
    //             NppStatus success1 = nppiFilterMedianGetBufferSize_8u_C1R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             // printf("%d, %d\n", sizeof(Npp8u), sizeof(scalar_t));
    //             // assert(sizeof(Npp8u) == sizeof(scalar_t));
    //             nppiFilterMedian_8u_C1R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
                
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_16u_C1R (reinterpret_cast<Npp16u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             NppStatus success1 = nppiFilterMedianGetBufferSize_16u_C1R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterMedian_16u_C1R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //         if (buffer_size != 0){
    //             cuda_mem_pool.free(buffer_size, buffer_tmp, out.device_id, out.event_sign);
    //         }
    //         if (z % OVERLAP_IMG_TIME == 0 && z != 0){
    //             cudaDeviceSynchronize();
    //         }
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }
    // template<typename scalar_t>
    // void median_filter_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     void* cSrc_tmp = nullptr;
        
    //     Npp32u buffer_size;
    //     for(int z = 0; z < pre_dim; ++z){
    //         void* buffer_tmp = nullptr;
    //         out_ptr = out.ptr + z * post_dim;
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         if(typeid(scalar_t) == typeid(uint8_t)){
                
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_8u_C3R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             NppStatus success1 = nppiFilterMedianGetBufferSize_8u_C3R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterMedian_8u_C3R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_16u_C3R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             NppStatus success1 = nppiFilterMedianGetBufferSize_16u_C3R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterMedian_16u_C3R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //         if (buffer_size != 0){
    //             cuda_mem_pool.free(buffer_size, buffer_tmp, out.device_id, out.event_sign);
    //         }
    //         if (z % OVERLAP_IMG_TIME == 0 && z != 0){
    //             cudaDeviceSynchronize();
    //         }
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }
    // template<typename scalar_t>
    // void median_filter_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     void* cSrc_tmp = nullptr;
        
    //     Npp32u buffer_size;
    //     for(int z = 0; z < pre_dim; ++z){
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         void* buffer_tmp = nullptr;
    //         out_ptr = out.ptr + z * post_dim;
    //         if(typeid(scalar_t) == typeid(uint8_t)){
                
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_8u_C4R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             NppStatus success1 = nppiFilterMedianGetBufferSize_8u_C4R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterMedian_8u_C4R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
                
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_16u_C4R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             NppStatus success1 = nppiFilterMedianGetBufferSize_16u_C4R(c_roi_npp, mask_npp, &buffer_size);
    //             if (buffer_size != 0){
    //                 mem_pool_alloc(buffer_size, buffer_tmp, out.device_id, out.stream_id);
    //             }
    //             Npp8u* buffer = reinterpret_cast<Npp8u*>(buffer_tmp);
                
    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterMedian_16u_C4R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp,
    //                 buffer
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //         if (buffer_size != 0){
    //             cuda_mem_pool.free(buffer_size, buffer_tmp, out.device_id, out.event_sign);
    //         }
    //         cudaDeviceSynchronize();
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }

    // template<typename scalar_t>
    // void box_filter_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         void* cSrc_tmp = nullptr;
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             // nppiCopyWrapBorder_8u_C1R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);
    //             nppiCopyConstBorder_8u_C1R(reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x, 0);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_8u_C1R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
                
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_16u_C1R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_16u_C1R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }

    // template<typename scalar_t>
    // void box_filter_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         void* cSrc_tmp = nullptr;
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             nppiCopyWrapBorder_8u_C3R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);
    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_8u_C3R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             nppiCopyWrapBorder_16u_C3R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);
    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_16u_C3R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }
    
    // template<typename scalar_t>
    // void box_filter_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, int pre_dim, int post_dim){
    //     NppiSize mask_npp = {std::get<1>(mask), std::get<0>(mask)};
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     scalar_t* out_ptr;
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }
    //     for(int z = 0; z < pre_dim; ++z){
    //         out_ptr = out.ptr + z * post_dim;
    //         void* cSrc_tmp = nullptr;
    //         mem_pool_alloc((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);
    //         if(typeid(scalar_t) == typeid(uint8_t)){
    //             Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_8u_C4R (reinterpret_cast<Npp8u*>(a_ptr + z * post_dim), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_8u_C4R(
    //                 reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp8u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else if(typeid(scalar_t) == typeid(uint16_t)){
                
    //             Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //             NppiSize c_roi_npp = {roi_npp.width + mask_npp.width, roi_npp.height + mask_npp.height};
    //             nppiCopyWrapBorder_16u_C4R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, (nstep_a + mask_npp.width) * sizeof(scalar_t), c_roi_npp, anchor_npp.y, anchor_npp.x);

    //             int new_nstep_a = nstep_a + mask_npp.width;
    //             nppiFilterBox_16u_C4R(
    //                 reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //                 (nstep_a + mask_npp.width) * sizeof(scalar_t),
    //                 reinterpret_cast<Npp16u*>(out_ptr),
    //                 nstep_out * sizeof(scalar_t),
    //                 roi_npp,
    //                 mask_npp,
    //                 anchor_npp
    //             );
    //         }
    //         else{
    //             std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //         }
    //         cudaDeviceSynchronize();
    //         cuda_mem_pool.free((roi_npp.width + mask_npp.width) * (roi_npp.height + mask_npp.height) * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    // }

    // template<typename scalar_t>
    // void conv_C1R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const Array<scalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& anchor, const std::tuple<int, int>& padding, int ndivisor){
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     NppiSize padding_npp = {std::get<1>(padding), std::get<0>(padding)};
    //     NppiSize kernel_size_npp = {std::get<1>(kernel_size), std::get<0>(kernel_size)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }

    //     void* cSrc_tmp = nullptr;
    //     NppiSize c_roi_npp = {roi_npp.height + padding_npp.height * 2, roi_npp.width + padding_npp.width * 2};
    //     mem_pool_alloc(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);

    //     if(typeid(scalar_t) == typeid(uint8_t)){
    //         Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //         nppiCopyConstBorder_8u_C1R (reinterpret_cast<Npp8u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_8u_C1R(
    //             reinterpret_cast<Npp8u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp8u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else if(typeid(scalar_t) == typeid(uint16_t)){
            
    //         Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //         nppiCopyConstBorder_16u_C1R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_16u_C1R(
    //             reinterpret_cast<Npp16u*>(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp16u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else{
    //         std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    //     cuda_mem_pool.free(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    // }

    // template<typename scalar_t>
    // void conv_C3R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const Array<scalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& anchor, const std::tuple<int, int>& padding, int ndivisor){
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     NppiSize padding_npp = {std::get<1>(padding), std::get<0>(padding)};
    //     NppiSize kernel_size_npp = {std::get<1>(kernel_size), std::get<0>(kernel_size)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }

    //     void* cSrc_tmp = nullptr;
    //     NppiSize c_roi_npp = {roi_npp.height + padding_npp.height * 2, roi_npp.width + padding_npp.width * 2};
    //     mem_pool_alloc(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);

    //     if(typeid(scalar_t) == typeid(uint8_t)){
    //         Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //         nppiCopyConstBorder_8u_C3R (reinterpret_cast<Npp8u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_8u_C3R(
    //             (Npp8u*)(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp8u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else if(typeid(scalar_t) == typeid(uint16_t)){
            
    //         Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //         nppiCopyConstBorder_16u_C3R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_16u_C3R(
    //             (Npp16u*)(cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x),
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp16u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else{
    //         std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    //     cuda_mem_pool.free(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    // }
    
    // template<typename scalar_t>
    // void conv_C4R(Array<scalar_t>& a_handle, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const Array<scalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& anchor, const std::tuple<int, int>& padding, int ndivisor){
    //     NppiPoint anchor_npp = {std::get<1>(anchor), std::get<0>(anchor)};
    //     NppiSize roi_npp = {std::get<1>(ROI), std::get<0>(ROI)};
    //     NppiSize padding_npp = {std::get<1>(padding), std::get<0>(padding)};
    //     NppiSize kernel_size_npp = {std::get<1>(kernel_size), std::get<0>(kernel_size)};
    
    //     Array<scalar_t> a(0, out.device_id);
    //     scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
    //     ewise_async_1op(a, out);
    //     if (nppGetStream() != streams[out.device_id][out.stream_id]){
    //         nppSetStream(streams[out.device_id][out.stream_id]);
    //     }

    //     void* cSrc_tmp = nullptr;
    //     NppiSize c_roi_npp = {roi_npp.height + padding_npp.height * 2, roi_npp.width + padding_npp.width * 2};
    //     mem_pool_alloc(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.stream_id);

    //     if(typeid(scalar_t) == typeid(uint8_t)){
    //         Npp8u* cSrc = reinterpret_cast<Npp8u*>(cSrc_tmp);
    //         nppiCopyConstBorder_8u_C4R (reinterpret_cast<Npp8u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_8u_C4R(
    //             (Npp8u*)cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x,
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp8u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else if(typeid(scalar_t) == typeid(uint16_t)){
            
    //         Npp16u* cSrc = reinterpret_cast<Npp16u*>(cSrc_tmp);
    //         nppiCopyConstBorder_16u_C4R (reinterpret_cast<Npp16u*>(a_ptr), nstep_a * sizeof(scalar_t), roi_npp, cSrc, c_roi_npp.width * sizeof(scalar_t), c_roi_npp, padding_npp.height, padding_npp.width, 0);

    //         int new_nstep_a = nstep_a + padding_npp.width;
    //         nppiFilter_16u_C4R(
    //             (Npp16u*)cSrc + anchor_npp.y * new_nstep_a + anchor_npp.x,
    //             (nstep_a + padding_npp.width * 2) * sizeof(scalar_t),
    //             (Npp16u*)out.ptr,
    //             nstep_out * sizeof(scalar_t),
    //             roi_npp,
    //             (Npp32s*)kernel.ptr,
    //             kernel_size_npp,
    //             anchor_npp,
    //             (Npp32s)ndivisor
    //         );
    //     }
    //     else{
    //         std::cerr << "The given datatype " + std::string(typeid(scalar_t).name()) + " is not supported for matrix_inv in the current version." << std::endl;
    //     }
    //     cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    //     cuda_mem_pool.free(c_roi_npp.width * c_roi_npp.height * sizeof(scalar_t), cSrc_tmp, out.device_id, out.event_sign);
    // }

    template<typename scalar_t, typename sscalar_t>
    void conv_1(Array<scalar_t>& a_handle, int nstep_a, int pre_dim, int post_dim, Array<scalar_t>& out, const Array<sscalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& padding, int stride, int dilation, int constant, int32_t offset_a, int32_t offset_k){  
        Array<scalar_t> a(0, out.device_id);
        scalar_t* a_ptr = cpy_gpus(a_handle, out, a);
        size_t block_num = ewise_async_2op(a, kernel, out);
        
        scalar_t* a_gpu, *out_gpu;
        
        MatrixShape kernel_shape(std::get<1>(kernel_size), std::get<0>(kernel_size));
        MatrixShape a_size(nstep_a, post_dim / nstep_a);
        MatrixShape padding_size(std::get<1>(padding), std::get<0>(padding));

        a_gpu = a_ptr;
        out_gpu = out.ptr;
        // state_check("0");
        convolution<scalar_t, sscalar_t><<<pre_dim, INIT_THREAD_NUM, 
        (CONV_INPUT_SHARED + kernel_shape.width * kernel_shape.height) * sizeof(scalar_t) + (INIT_THREAD_NUM / kernel_shape.width + 1) * sizeof(float) + sizeof(double) * 4,
         streams[out.device_id][out.stream_id]>>>(a_gpu, kernel.ptr, out_gpu, kernel_shape, a_size, padding_size, dilation, stride, constant, offset_a, offset_k, 1);

        // state_check("1");
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
    }
}
}

template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_IMGSPROC(py::module& m){
    using namespace gpu;
    using namespace pygp_img;
    /*image operations*/
    // m.def("gaussian_filter", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, uint8_t type, int pre_dim, int post_dim){
    //     switch(type){
    //         case 1:
    //             gaussian_filter_C1R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         case 3:
    //             gaussian_filter_C3R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         case 4:
    //             gaussian_filter_C4R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });

    // m.def("laplacian_filter", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, int mask, uint8_t type, int pre_dim, int post_dim){
    //     switch(type){
    //         case 1:
    //             laplacian_filter_C1R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         case 3:
    //             laplacian_filter_C3R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         case 4:
    //             laplacian_filter_C4R(a, nstep_a, out, nstep_out, ROI, mask, pre_dim, post_dim);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });

    // m.def("sobel_filter", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, bool horiz, uint8_t type, int pre_dim, int post_dim){
    //     switch(type){
    //         case 1:
    //             sobel_filter_C1R(a, nstep_a, out, nstep_out, ROI, pre_dim, post_dim, horiz);
    //             break;
    //         case 3:
    //             sobel_filter_C3R(a, nstep_a, out, nstep_out, ROI, pre_dim, post_dim, horiz);
    //             break;
    //         case 4:
    //             sobel_filter_C4R(a, nstep_a, out, nstep_out, ROI, pre_dim, post_dim, horiz);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });

    // m.def("box_filter", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, uint8_t type, int pre_dim, int post_dim){
    //     switch(type){
    //         case 1:
    //             box_filter_C1R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         case 3:
    //             box_filter_C3R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         case 4:
    //             box_filter_C4R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });

    // m.def("median_filter", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const std::tuple<int, int>& mask, const std::tuple<int, int>& anchor, uint8_t type, int pre_dim, int post_dim){
    //     switch(type){
    //         case 1:
    //             median_filter_C1R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         case 3:
    //             median_filter_C3R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         case 4:
    //             median_filter_C4R(a, nstep_a, out, nstep_out, ROI, mask, anchor, pre_dim, post_dim);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });
    
    // m.def("conv_2D", [](Array<scalar_t>& a, int nstep_a, Array<scalar_t>& out, int nstep_out, const std::tuple<int, int>& ROI, const Array<scalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& anchor, const std::tuple<int, int>& padding, int ndivisor, uint8_t type){
    //     switch(type){
    //         case 1:
    //             conv_C1R(a, nstep_a, out, nstep_out, ROI, kernel, kernel_size, anchor, padding, ndivisor);
    //             break;
    //         case 3:
    //             conv_C3R(a, nstep_a, out, nstep_out, ROI, kernel, kernel_size, anchor, padding, ndivisor);
    //             break;
    //         case 4:
    //             conv_C4R(a, nstep_a, out, nstep_out, ROI, kernel, kernel_size, anchor, padding, ndivisor);
    //             break;
    //         default:
    //             std::cerr << "The support channel is 1, 3, 4, the current channel num is: " << type << std::endl;
    //     }
    // });

    
    m.def("conv", [](Array<scalar_t>& a, int nstep_a, int pre_dim, int post_dim, Array<scalar_t>& out, const Array<sscalar_t>& kernel, const std::tuple<int, int>& kernel_size, const std::tuple<int, int>& padding, int stride, int dilation, int constant, int32_t offset_a, int32_t offset_k){ 
        conv_1(a, nstep_a, pre_dim, post_dim, out, kernel, kernel_size, padding, stride, dilation, constant, offset_a, offset_k);
    });


    m.def("SIFT", [](){

    });

    m.def("uLBP", [](){

    });

    m.def("HOG", [](){});
    m.def("Hist", [](){});
    m.def("DIF", [](){});
}

template<typename scalar_t>
void TEMPLATE_BIND_ARRAY(py::module& m){
    // printf("%s\n",("Array_" + std::string(typeid(scalar_t).name())).c_str());
    py::class_<Array<scalar_t>>(m, ("Array_" + std::string(typeid(scalar_t).name())).c_str())
        .def(py::init<>(), py::return_value_policy::take_ownership)
        .def(py::init<const size_t, int>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &Array<scalar_t>::size)
        .def_readonly("stream_id", &Array<scalar_t>::stream_id)
        .def_readonly("dev_id", &Array<scalar_t>::device_id)
        .def("ptr", &Array<scalar_t>::ptr_as_int);
    
    m.def("to_numpy", [](Array<scalar_t>& a, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset){
        /* [ ] TODO: The following three rows should be replaced after using async alloc
                        by: Scalar_t* Array<scalar_t> = a_ptr;
        */
        // if(a.stream_id != -1){
        //     cudaEventSynchronize(a.event_sign);
        // }
        if(a.stream_id == -1) a.stream_id = rand() % STREAM_NUM_NDARRAY;
        int shape_size = 1;
        for(int i = 0; i < shape.size(); ++i){
                shape_size *= shape[i];
        }
        const size_t ELEM_SIZE = sizeof(scalar_t);
        scalar_t* array = new scalar_t[shape_size];
        cudaMemcpyAsync(array, a.ptr + offset, shape_size * ELEM_SIZE, cudaMemcpyDeviceToHost, streams[a.device_id][a.stream_id]);
        cudaStreamSynchronize(streams[a.device_id][a.stream_id]);
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(), [](size_t& c){return c * ELEM_SIZE;} );
        
        return py::array_t<scalar_t>(shape, numpy_strides, array);
    });

    m.def("from_numpy", [](py::array_t<scalar_t>& a, Array<scalar_t>* out){
            
        if(out->stream_id == -1) out->stream_id = rand() % STREAM_NUM_NDARRAY;

        const size_t ELEM_SIZE = sizeof(scalar_t);
        // std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
        cudaError_t err = cudaMemcpyAsync(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice, streams[out->device_id][out->stream_id]);
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    });

}

template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_FUNCS_DIM1(py::module& m){
    
    using namespace gpu;
    using namespace pygp_tensor;

    m.def("ewise_sin", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 0);
    });
    m.def("ewise_cos", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 1);
    });
    m.def("ewise_tan", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 2);
    });
    m.def("ewise_sqrt", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 3);
    });
    m.def("ewise_arcsin", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 4);
    });
    m.def("ewise_arccos", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 5);
    });
    m.def("ewise_arctan", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 6);
    });
    m.def("ewise_sign", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 7);
    });
    m.def("ewise_exp", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 8);
    });
    m.def("ewise_abs", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 9);
    });
    m.def("ewise_neg", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 10);
    });
    m.def("ewise_ceil", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 11);
    });
    m.def("ewise_floor", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 12);
    });
    m.def("ewise_loge", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 13);
    });
    m.def("ewise_log2", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 14);
    });
    m.def("ewise_log10", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 15);
    });

    m.def("ewise_sum", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 0);
    });
    m.def("ewise_min", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 1);
    });
    m.def("ewise_max", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 2);
    });
    m.def("ewise_mean", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 3);
    });
    m.def("ewise_argmax", [](const Array<scalar_t>& a, Array<int32_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, int32_t>(a, out, pre_dim, post_dim, offset_a, 4);
    });
    m.def("ewise_argmin", [](const Array<scalar_t>& a, Array<int32_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, int32_t>(a, out, pre_dim, post_dim, offset_a, 5);
    });
    m.def("ewise_std", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 6);
    });
    m.def("ewise_var", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 7);
    });
    m.def("ewise_cumsum", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 8);
    });
    m.def("ewise_cumprob", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int pre_dim=1, int post_dim, int32_t offset_a){
        oper_dim_1<scalar_t, sscalar_t>(a, out, pre_dim, post_dim, offset_a, 9);
    });

    
    m.def("matrix_T", [](const Array<scalar_t>& a, Array<scalar_t>& out, int pre_dim, int col_num, int post_dim, int offset){
    matrix_transpose<scalar_t>(a, out, pre_dim, col_num, post_dim, offset);
    });
    
    m.def("matrix_inv", [](Array<scalar_t>& a, Array<scalar_t>& out, int pre_dim, int col_num, int post_dim, int offset, Array<scalar_t>& infos){
    
    matrix_inv<scalar_t>(a, out, pre_dim, col_num, post_dim, offset, infos);
    });
    
    m.def("matrix_det", [](Array<scalar_t>& a, Array<scalar_t>& out, int pre_dim, int col_num, int post_dim, int offset, Array<scalar_t>& infos){
    matrix_det<scalar_t>(a, out, pre_dim, col_num, post_dim, offset, infos);
    });

    m.def("matrix_diagonal_sum", [](Array<scalar_t>& a, Array<scalar_t>& out, int pre_dim, int col_num, int offset){
    matrix_diagonal_sum<scalar_t>(a, out, pre_dim, col_num, offset);
    });

    
    m.def("transfer_1", [](Array<scalar_t>& out, Array<scalar_t>& in_handle, const std::vector<int32_t>& idxs, int unit_len, int offset){

        Array<scalar_t> in(0, out.device_id);
        scalar_t* in_ptr = cpy_gpus(in_handle, out, in);
        ewise_async_1op(in, out);
        int thread_num = 1024;
        int unit_total = unit_len * idxs.size();
        if (unit_total < thread_num){
            thread_num = unit_total + 32 - unit_total % 32;
        }
        int block_num = (unit_total + thread_num - 1) / thread_num;
        void* idxs_gpu_tmp;
        mem_pool_alloc(sizeof(int32_t) * idxs.size(), idxs_gpu_tmp, out.device_id, out.stream_id);
        int32_t* idxs_gpu = reinterpret_cast<int32_t*>(idxs_gpu);
        cudaMalloc(&idxs_gpu, sizeof(int32_t) * idxs.size());
        cudaMemcpyAsync(idxs_gpu, &(idxs[0]), sizeof(int32_t) * idxs.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        // printf("%d, %d, %d, %d, %d\n", block_num, thread_num, unit_len, offset, idxs.size());
        transfer_1<scalar_t><<<block_num, thread_num, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, in.ptr, idxs_gpu, idxs.size(), unit_len, offset);
        
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
        cuda_mem_pool.free(sizeof(int32_t) * idxs.size(), idxs_gpu_tmp, out.device_id, out.event_sign);

    });

    
    m.def("_old_transfer", [](Array<scalar_t>& out, Array<scalar_t>& in_handle, const std::vector<int32_t>& unit_idxs, int unit_len, int offset, int type){

        Array<scalar_t> in(0, out.device_id);
        scalar_t* in_ptr = cpy_gpus(in_handle, out, in);
        ewise_async_1op(in, out);
        int thread_num = INIT_THREAD_NUM;
        int unit_total = unit_idxs.size();
        if (unit_total < thread_num){
            thread_num = unit_total + 32 - unit_total % 32;
        }
        int block_num = (unit_total + thread_num - 1) / thread_num;
        void* idxs_gpu_tmp;
        mem_pool_alloc(sizeof(int32_t) * unit_idxs.size(), idxs_gpu_tmp, out.device_id, out.stream_id);
        int32_t* idxs_gpu = reinterpret_cast<int32_t*>(idxs_gpu_tmp);
        cudaMemcpyAsync(idxs_gpu, &(unit_idxs[0]), sizeof(int32_t) * unit_idxs.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        // printf("%d, %d, %d, %d, %d\n", block_num, thread_num, unit_idxs.size(), offset, unit_idxs[0]);
        if (type == 0){
            old_transfer_get<scalar_t><<<block_num, thread_num, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, unit_idxs.size(), unit_len, offset);
        }
        else{
            old_transfer_set<scalar_t><<<block_num, thread_num, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, unit_idxs.size(), unit_len, offset);
        }
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
        cuda_mem_pool.free(sizeof(int32_t) * unit_idxs.size(), idxs_gpu_tmp, out.device_id, out.event_sign);

    });

    
    m.def("transfer", [](Array<scalar_t>& out, Array<scalar_t>& in_handle, const std::vector<int32_t>& unit_idxs, const std::vector<int32_t>& unit_sizes, int32_t unit_len, int in_offset, int out_offset, int type){

        Array<scalar_t> in(0, out.device_id);
        scalar_t* in_ptr = cpy_gpus(in_handle, out, in);
        int thread_num = INIT_THREAD_NUM;
        int unit_total = unit_idxs.size();
        if (unit_total < thread_num){
            thread_num = unit_total + 32 - unit_total % 32;
        }
        int block_num = (unit_total + thread_num - 1) / thread_num;
        void* idxs_gpu_tmp, *sizes_gpu_tmp;
        mem_pool_alloc(sizeof(int32_t) * unit_idxs.size(), idxs_gpu_tmp, out.device_id, out.stream_id);
        mem_pool_alloc(sizeof(int32_t) * unit_sizes.size(), sizes_gpu_tmp, out.device_id, out.stream_id);
        int32_t* idxs_gpu = reinterpret_cast<int32_t*>(idxs_gpu_tmp);
        int32_t* sizes_gpu = reinterpret_cast<int32_t*>(sizes_gpu_tmp);
        cudaMemcpyAsync(idxs_gpu, &(unit_idxs[0]), sizeof(int32_t) * unit_idxs.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        cudaMemcpyAsync(sizes_gpu, &(unit_sizes[0]), sizeof(int32_t) * unit_sizes.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        // printf("%d, %d, %d, %d, %d\n", block_num, thread_num, unit_idxs.size(), offset, unit_idxs[0]);
        if (type == 0){
            transfer_get<scalar_t><<<block_num, thread_num, unit_sizes.size() * sizeof(int32_t), streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, sizes_gpu, unit_len, in_offset, out_offset, unit_sizes.size());
        }
        else{
            transfer_set<scalar_t><<<block_num, thread_num, unit_sizes.size() * sizeof(int32_t), streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, sizes_gpu, unit_len, in_offset, out_offset, unit_sizes.size());
        }
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
        cuda_mem_pool.free(sizeof(int32_t) * unit_idxs.size(), idxs_gpu_tmp, out.device_id, out.event_sign);

    });

    
    m.def("get_properties", &get_properties);


    m.def("check", [](){
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    });
    m.def("compact", Compact<scalar_t>);
    m.def("ewise_setitem", EwiseSetitem<scalar_t>);
    m.def("scalar_setitem", ScalarSetitem<scalar_t>);
    m.def("concatenate", Concatenate<scalar_t>);
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void TEMPLATE_BIND_FUNCS(py::module& m){
    
    using namespace gpu;
    using namespace pygp_tensor;

    m.def("wait", [](Array<scalar_t>& out){
        if (out.stream_id != -1){
            cudaStreamWaitEvent(streams[out.device_id][out.stream_id], out.event_sign, 0);
        }
    });
    m.def("cc", [](){
        // ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 0);
    });
    m.def("ewise_add", [](Array<scalar_t>& a, Array<sscalar_t>& b, Array<tscalar_t>& out, int32_t& offset_a, int32_t& offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 0);
    });
    m.def("ewise_sub", [](Array<scalar_t>& a, Array<sscalar_t>& b, Array<tscalar_t>& out, int32_t& offset_a, int32_t& offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 1);
    });
    m.def("ewise_mul", [](Array<scalar_t>& a, Array<sscalar_t>& b, Array<tscalar_t>& out, int32_t& offset_a, int32_t& offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 2);
    });
    m.def("ewise_div", [](Array<scalar_t>& a, Array<sscalar_t>& b, Array<tscalar_t>& out, int32_t& offset_a, int32_t& offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 3);
    });
    m.def("ewise_pow", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, tscalar_t>(a, b, out, offset_a, offset_b, 4);
    });
    
    m.def("ewise_le", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 5);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("ewise_lt", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 6);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("ewise_ge", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 7);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("ewise_gt", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 8);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("ewise_ne", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 9);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    
    m.def("ewise_eq", [](const Array<scalar_t>& a, const Array<scalar_t>& b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        ewise_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, offset_b, 10);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    //     m.def("ewise_eq", [](const Array<scalar_t>& a, const Array<bool>& b, bool& sign)){
    //         scalar_compute_2op(a, b, , 4);
    //     }

    

    m.def("ewise_sub_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 1);
    });
    m.def("ewise_add_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 0);
    });
    m.def("ewise_mul_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 2);
    });
    m.def("ewise_div_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 3);
    });


    m.def("scalar_add", [](const Array<scalar_t>& a, scalar_t b, Array<scalar_t>& out, int offset_a){
        scalar_compute_2op<scalar_t>(a, b, out, offset_a, 0);
    });
    m.def("scalar_sub", [](const Array<scalar_t>& a, scalar_t b, Array<scalar_t>& out, int offset_a){
        scalar_compute_2op<scalar_t>(a, b, out, offset_a, 1);
    });
    m.def("scalar_mul", [](const Array<scalar_t>& a, scalar_t b, Array<scalar_t>& out, int offset_a){
        scalar_compute_2op<scalar_t>(a, b, out, offset_a, 2);
    });
    m.def("scalar_div", [](const Array<scalar_t>& a, scalar_t b, Array<scalar_t>& out, int offset_a){
        scalar_compute_2op<scalar_t>(a, b, out, offset_a, 3);
    });
    m.def("scalar_pow", [](const Array<scalar_t>& a, scalar_t b, Array<scalar_t>& out, int offset_a){
        scalar_compute_2op<scalar_t>(a, b, out, offset_a, 4);
    });
    
    m.def("scalar_le", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 5);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("scalar_lt", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 6);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("scalar_ge", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 7);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("scalar_gt", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 8);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("scalar_ne", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 9);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });
    m.def("scalar_eq", [](const Array<scalar_t>& a, sscalar_t b, Array<bool>& out, int32_t offset_a, int32_t offset_b){
        scalar_compute_2op<scalar_t, sscalar_t, bool>(a, b, out, offset_a, 10);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
    });

    m.def("ewise_assign", [](Array<scalar_t>& out, scalar_t val, int post_dim, int offset_a){
        ewise_assign<scalar_t>(out, val, post_dim, offset_a);
    });
    m.def("where", [](const Array<bool>& a, const Array<scalar_t>& b, const Array<sscalar_t>& c, Array<tscalar_t>& out, size_t size,
    int offset_a, int offset_b, int offset_c){
        ewise_compute_3op<bool, scalar_t, sscalar_t, tscalar_t>(a, b, c, out, offset_a, offset_b, offset_c, 0);
    });

    /*Matrix operations*/
    m.def("matrix_dot", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>& out,
     int pre_dim, int col_num_a, int col_num_b, int post_dim_a, int post_dim_b, 
     int offset_a, int offset_b){
        matrix_opers<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim, col_num_a, col_num_b, post_dim_a, post_dim_b, 
     offset_a, offset_b);
     });

}

PYBIND11_MODULE(ndarray_cuda_backend, m){

    handle_create();
    streams_create(std::vector<int>(1, 0));
    m.def("set_device", &streams_create);
    m.def("is_available", &is_available);
    m.def("cuda_mem_available", &cuda_mem_available);
    m.def("cuda_mem_total", &cuda_mem_total);
    m.def("cuda_mem_ratio", &cuda_mem_ratio);

    TEMPLATE_BIND_ARRAY<bool>(m);
    TEMPLATE_BIND_ARRAY<uint8_t>(m);
    TEMPLATE_BIND_ARRAY<uint16_t>(m);
    TEMPLATE_BIND_ARRAY<uint32_t>(m);
    TEMPLATE_BIND_ARRAY<uint64_t>(m);
    TEMPLATE_BIND_ARRAY<int8_t>(m);
    TEMPLATE_BIND_ARRAY<int16_t>(m);
    TEMPLATE_BIND_ARRAY<int32_t>(m);
    TEMPLATE_BIND_ARRAY<int64_t>(m);
    TEMPLATE_BIND_ARRAY<float>(m);
    TEMPLATE_BIND_ARRAY<double>(m);

    TEMPLATE_BIND_FUNCS<uint8_t, uint8_t, uint8_t>(m);
    TEMPLATE_BIND_FUNCS<uint16_t, uint16_t, uint16_t>(m);
    TEMPLATE_BIND_FUNCS<uint32_t, uint32_t, uint32_t>(m);
    TEMPLATE_BIND_FUNCS<uint64_t, uint64_t, uint64_t>(m);
    TEMPLATE_BIND_FUNCS<int8_t, int8_t, int8_t>(m);
    TEMPLATE_BIND_FUNCS<int16_t, int16_t, int16_t>(m);
    TEMPLATE_BIND_FUNCS<int32_t, int32_t, int32_t>(m);
    TEMPLATE_BIND_FUNCS<int64_t, int64_t, int64_t>(m);
    TEMPLATE_BIND_FUNCS<float, float, float>(m);
    TEMPLATE_BIND_FUNCS<double, double, double>(m);

    TEMPLATE_BIND_FUNCS_DIM1<uint8_t, uint8_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint16_t, uint16_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint32_t, uint32_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint64_t, uint64_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int8_t, int8_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int16_t, int16_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int32_t, int32_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int64_t, int64_t>(m);
    TEMPLATE_BIND_FUNCS_DIM1<float, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<double, double>(m);

    TEMPLATE_BIND_FUNCS_DIM1<uint8_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint16_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint32_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint64_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int8_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int16_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int32_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int64_t, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<float, float>(m);
    TEMPLATE_BIND_FUNCS_DIM1<double, float>(m);

    TEMPLATE_BIND_FUNCS_DIM1<uint8_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint16_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint32_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<uint64_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int8_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int16_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int32_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<int64_t, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<float, double>(m);
    TEMPLATE_BIND_FUNCS_DIM1<double, double>(m);

    TEMPLATE_BIND_IMGSPROC<uint8_t, uint8_t>(m);
    TEMPLATE_BIND_IMGSPROC<uint16_t, uint16_t>(m);
    TEMPLATE_BIND_IMGSPROC<uint32_t, uint32_t>(m);
    TEMPLATE_BIND_IMGSPROC<uint64_t, uint64_t>(m);
    TEMPLATE_BIND_IMGSPROC<int8_t, int8_t>(m);
    TEMPLATE_BIND_IMGSPROC<int16_t, int16_t>(m);
    TEMPLATE_BIND_IMGSPROC<int32_t, int32_t>(m);
    TEMPLATE_BIND_IMGSPROC<int64_t, int64_t>(m);
    TEMPLATE_BIND_IMGSPROC<float, float>(m);
    TEMPLATE_BIND_IMGSPROC<double, double>(m);
    
    TEMPLATE_BIND_IMGSPROC<uint8_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<uint16_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<uint32_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<uint64_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<int8_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<int16_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<int32_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<int64_t, float>(m);
    TEMPLATE_BIND_IMGSPROC<float, float>(m);
    TEMPLATE_BIND_IMGSPROC<double, float>(m);
    
    TEMPLATE_BIND_IMGSPROC<uint8_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<uint16_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<uint32_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<uint64_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<int8_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<int16_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<int32_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<int64_t, double>(m);
    TEMPLATE_BIND_IMGSPROC<float, double>(m);
    TEMPLATE_BIND_IMGSPROC<double, double>(m);
}