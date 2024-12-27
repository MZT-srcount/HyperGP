#include <cuda_runtime.h>
#include "cuda.h"
#include <cublas_v2.h>
#include <unordered_map>


#define STREAM_NUM_NDARRAY 12
#define DEAULT_ELEMENT_SIZE 1 // the default create num each time
#define MAX_ELEMENT_SIZE 4   // max remain size when free(each elem size)
#define OVERLAP_IMG_TIME 10
std::unordered_map<int, cudaStream_t*> streams;


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

size_t cuda_mem_available(int device_id){
    size_t avail(0), total(0);
    CHECK(cudaMemGetInfo(&avail,&total));
    return avail;
}

size_t cuda_mem_total(int device_id){
    size_t avail(0), total(0);
    CHECK(cudaMemGetInfo(&avail,&total));
    return total;
}

float cuda_mem_ratio(){
    size_t avail(0), total(0);
    CHECK(cudaMemGetInfo(&avail,&total));
    return float(avail) / total;
}

void state_check(std::string ad_info){
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("%s\n", ((ad_info + std::string("-") + cudaGetErrorString(err))).c_str());
        throw std::runtime_error(cudaGetErrorString(err));
        printf("here????????23123\n");
        exit(-1);
    }
    else{
        printf("Succeed!! %s \n", ad_info.c_str());
    }
}

bool is_available(int device_id) {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) {
        std::cerr << "Error: Unable to query device count: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
 
    if (device_id >= count) {
        std::cerr << "Error: Device " << device_id << " is not available" << std::endl;
        return false;
    }
 
    cudaSetDevice(device_id);
    cudaError_t set_device_error = cudaGetLastError();
    if (set_device_error != cudaSuccess) {
        std::cerr << "Error: Unable to set device " << device_id << ": " << cudaGetErrorString(set_device_error) << std::endl;
        return false;
    }
 
    return true;
}
