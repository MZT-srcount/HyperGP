#include <cuda_runtime.h>
#include "cuda_mem_manager.h"

template<typename scalar_t>
struct Array{

    Array() {
        cudaEventCreate(&this->event_sign);
        /* to avoid out of video memory, the array still store in CPU memory */
        // ptr = new scalar_t[size];
    }
    // Array(const Array<scalar_t>& other){
    //     // printf("!!!!????\n");
    //     this->stream_id = other.stream_id;
    //     this->device_id = other.device_id;
    //     cudaEventCreate(&this->event_sign);
    //     this->ptr = other.ptr;
    //     this->size = other.size;
    // }
    Array(const size_t size, int device_id) {
        /* to avoid out of video memory, the array still store in CPU memory */
        // ptr = new scalar_t[size];
        CHECK(cudaSetDevice(device_id));
        cudaEventCreate(&this->event_sign);
        this->size = size;
        this->device_id = device_id;
        if(size > 0){
            void* tmp_ptr = (void*) this->ptr;
            EventState old_event;
            bool suc = cuda_mem_pool.alloc(size * sizeof(scalar_t), tmp_ptr, this->device_id, old_event);
            if (old_event.event_sign){
                cudaEventDestroy(this->event_sign);
                this->event_sign = old_event.event;
            }
            this->ptr = reinterpret_cast<scalar_t*>(tmp_ptr);
            if (suc == 0) throw std::runtime_error("Can not alloc cuda memory\n");
        }
        // cudaError_t err = cudaMalloc(&(this->ptr), size * ELEM_SIZE);
        // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
        this->device_id = device_id;
    }
    void init(const size_t size, int device_id){
        /* to avoid out of video memory, the array still store in CPU memory */
        // ptr = new scalar_t[size];
        // printf("here???????\n");
        CHECK(cudaSetDevice(device_id));
        cudaEventCreate(&this->event_sign);
        this->size = size;
        this->device_id = device_id;
        if(size > 0){
            void* tmp_ptr = (void*) this->ptr;
            
            EventState old_event;
            bool suc = cuda_mem_pool.alloc(size * sizeof(scalar_t), tmp_ptr, this->device_id, old_event);
            if (old_event.event_sign){
                cudaEventDestroy(this->event_sign);
                this->event_sign = old_event.event;
            }
            this->ptr = reinterpret_cast<scalar_t*>(tmp_ptr);
            if (suc == 0) throw std::runtime_error("Can not alloc cuda memory\n");
        }
        // cudaError_t err = cudaMalloc(&(this->ptr), size * ELEM_SIZE);
        // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    ~Array(){
        //[ ] TODO: time to destroy event
        // cudaEventDestroy(this->event_sign);
        cuda_mem_pool.free(this->size * sizeof(scalar_t), (void*)this->ptr, this->device_id, this->event_sign);
        // cudaError_t err = cudaFree(this->ptr);
        // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    }
    Array<scalar_t>& operator =(const Array<scalar_t>& other){
        // printf("here????\n");
        this->stream_id = other.stream_id;
        this->device_id = other.device_id;
        this->event_sign = other.event_sign;
        this->size = other.size;
        
        if(this->size > 0){
            void* tmp_ptr = (void*) this->ptr;
            EventState old_event;
            bool suc = cuda_mem_pool.alloc(size * sizeof(scalar_t), tmp_ptr, this->device_id, old_event);
            if (old_event.event_sign){
                cudaEventDestroy(this->event_sign);
                this->event_sign = old_event.event;
            }
            this->ptr = reinterpret_cast<scalar_t*>(tmp_ptr);
            if (suc == 0) throw std::runtime_error("Can not alloc cuda memory\n");
        }
        return *this;
    }
    /*[ ] TODO warning: This stream should be kept until all compute work be done, pay attention to the ~Array function to automatic deallocate the stream*/
    
    int stream_id = -1;
    int device_id;
    cudaEvent_t event_sign;
    scalar_t* ptr;
    size_t size = 0;
    size_t ptr_as_int() { return (size_t)ptr; }

};

Array<float> array_float(int dev_id, const size_t size=0){
    if (size == 0){
        return Array<float>();
    }
    return Array<float>(size, dev_id);
}

Array<double> array_double(int dev_id, const size_t size=0){
    if (size == 0){
        return Array<double>();
    }
    return Array<double>(size, dev_id);
}

Array<int32_t> array_int32(int dev_id, const size_t size=0){
    if (size == 0){
        return Array<int32_t>();
    }
    return Array<int32_t>(size, dev_id);
}

Array<int64_t> array_int64(int dev_id, const size_t size=0){
    if (size == 0){
        return Array<int64_t>();
    }
    return Array<int64_t>(size, dev_id);
}

Array<int8_t> array_int8(int dev_id, const size_t size=0){
    if (size == 0){
        return Array<int8_t>();
    }
    return Array<int8_t>(size, dev_id);
}