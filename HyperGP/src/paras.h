#ifndef PARAS_H
#define PARAS_H
#include <cuda_runtime.h>
#include "cuda.h"
#include <cublas_v2.h>
#include <unordered_map>
#include <iostream>
#include <vector>


#define STREAM_NUM_NDARRAY 6
#define STREAM_NUM_EXEC 3
#define DEAULT_ELEMENT_SIZE 1 // the default create num each time
#define MAX_ELEMENT_SIZE 4   // max remain size when free(each elem size)
#define OVERLAP_IMG_TIME 10
// extern std::unordered_map<int, cudaStream_t*> streams;

std::unordered_map<int, cudaStream_t*> streams = {
    {0, new cudaStream_t[STREAM_NUM_NDARRAY]}
};

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



void check_sync(){
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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
void streams_create(const std::vector<int>& devices, int stream_num){
    for(int i = 0; i < devices.size(); ++i){
        if(streams.find(devices[i]) == streams.end()){
            if (is_available(devices[i])){
                CHECK(cudaSetDevice(devices[i]));
                streams[devices[i]] = new cudaStream_t[stream_num];
                for(int k = 0; k < stream_num; ++k){
                    cudaStreamCreate(&streams[devices[i]][k]);
                }
            }
        }
    }
}

struct GlobalStreams{
    // GlobalStreams(std::unordered_map<int, cudaStream_t*>& streams){
    //     this->streams = streams;
    // }
    void init(std::unordered_map<int, cudaStream_t*>& streams){
        this->streams = streams;
    }

    std::unordered_map<int, cudaStream_t*> streams;
};
// extern "C" std::unordered_map<int, cudaStream_t*>& get_streams() {
//     return streams;
// }

#endif

