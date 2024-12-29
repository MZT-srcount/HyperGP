#pragma once
#include "cuda_runtime_api.h"
#include <iostream>
#include <list>
#include <stdint.h>
#include <mutex>
#include <unordered_map>
#include "paras.h"

struct MemUnit{
    MemUnit(void* ptr, const cudaEvent_t& event){
        this->ptr = ptr;
        this->event = event;
    }
    MemUnit(void* ptr, bool sign){
        this->ptr = ptr;
        this->event_sign = false;
    }
    void* ptr;
    cudaEvent_t event;
    bool event_sign = true;
};

struct EventState{
    EventState(const cudaEvent_t& event){
        this->event = event;
        this->event_sign = true;
    }
    EventState(){
        this->event_sign = false;
    }
    cudaEvent_t event;
    bool event_sign;
};

class CudaMemoryPool
{
public:
    // 获取内存池单例
    static CudaMemoryPool &instance()
    {
        static CudaMemoryPool ins;
        return ins;
    }

    bool alloc(size_t elementSize, void*&element, int device, EventState& event, int stream_id = -1)
    {
        CHECK(cudaSetDevice(device));
        // std::unique_lock<std::mutex> locker(m_mtx);
        if (m_elementListMap[device].count(elementSize) == 0)
        {
            std::list<MemUnit> elementList;
            uint32_t ret = create(elementList, elementSize, DEAULT_ELEMENT_SIZE);

            // std::cerr << "00element create " << ret << " success, elementSize: " << elementSize << std::endl;
            if (ret == 0){
                std::vector<size_t> keys;
                for (const auto& kv : m_elementListMap[device]) {
                    keys.push_back(kv.first);
                }
                if(keys.size() == 0){
                    return false;
                }
                std::sort(keys.begin(), keys.end(), [](const size_t& a, const size_t& b){
                    return a > b;
                });
                size_t free_size = -1;
                for(int i = 1; i < keys.size(); ++i){
                    if(keys[i] > elementSize && m_elementListMap[device][keys[i]].size() > 0){
                        free_size = keys[0];
                    }
                    if(keys[i] < elementSize){
                        break;
                    }
                }
                if (free_size != -1){
                    MemUnit elem = m_elementListMap[device][free_size].front();
                    free(free_size, elem.ptr, device, elem.event, true);
                }
                else{
                    return false;
                }
            }
            if(create(elementList, elementSize, DEAULT_ELEMENT_SIZE) == 0){
                return false;
            }

            m_elementListMap[device].insert({elementSize, elementList});
        }

        auto &elementList = m_elementListMap[device][elementSize];
        if (elementList.empty()) // if list is empty, rebuild
        {
            uint32_t ret = create(elementList, elementSize, DEAULT_ELEMENT_SIZE);
            if (ret == 0) // create fail..
            {
                std::cerr << "create  " << elementSize << " elements all failed!" << std::endl;
                return false;
            }
        }
        MemUnit elem = elementList.front();
        elementList.pop_front();
        if (elem.ptr == 0)
        {
            std::cerr << "pop element is null!" << std::endl;
            return false;
        }
        if (elem.event_sign){
            event = EventState(elem.event);
            if (stream_id != -1){
                cudaStreamWaitEvent(streams[device][stream_id], event.event, 0);
            }
        }
        element = elem.ptr;
        return true;
    }

    // 回收内存，链表中元素个数超过一定数量时释放内存
    void free(size_t elementSize, void*element, int device, cudaEvent_t& event, bool must_free=false)
    {
        if (element == nullptr || elementSize == 0)
            return;
        CHECK(cudaSetDevice(device));

        // std::unique_lock<std::mutex> locker(m_mtx);
        cudaError_t err;
        if (m_elementListMap[device].count(elementSize) > 0)
        {
            auto &elementList = m_elementListMap[device][elementSize];
            // if (mem_available_ratio() < 0.5){
            //     if (mem_available_ratio() < 0.1){
            //         this->max_elem_size = 1;
            //     }
            //     else{
            //         this->max_elem_size = this->max_elem_size / 2 > 2 ? this->max_elem_size / 2 : 2;
            //     }
            // }
            // else{
            //     this->max_elem_size = MAX_ELEMENT_SIZE;
            // }
            if (elementList.size() > this->max_elem_size || must_free){// 超过一定数量时释放
                err = cudaFree(element);
                // cudaEventDestroy(event);
                if (err != cudaSuccess) throw std::runtime_error((std::string("here????? ") + cudaGetErrorString(err)).c_str());
            }
            else{
                elementList.push_back(MemUnit(element, event));//cudaStreamWaitEvent(streams[out.device_id][out.stream_id], a.event_sign, 0)
            }
        }
        else // 如果找不到直接释放
        {
            err = cudaFree(element);
            // cudaEventDestroy(event);
            if (err != cudaSuccess) throw std::runtime_error((std::string("here free? ") + cudaGetErrorString(err)).c_str());
        }
    }

    // 释放内存池中所有内存
    void destroy()
    {
        // std::unique_lock<std::mutex> locker(m_mtx);
        for(auto &stream: m_elementListMap){
            cudaSetDevice(stream.first);//CHECK(cudaSetDevice(stream.first));
            for (auto &it : stream.second)
            {
                auto &elementList = it.second;
                cudaError_t err;
                for (MemUnit element: elementList){

                    err = cudaFree(element.ptr);
                    if (element.event_sign){
                        cudaEventDestroy(element.event);
                    }
                    // if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
                }
            }
        }
    }
    // 禁止复制和赋值操作
    // CudaMemoryPool() = delete;
    ~CudaMemoryPool()
    {
        destroy();
    }
    int max_elem_size = MAX_ELEMENT_SIZE;
    // CudaMemoryPool(const CudaMemoryPool &other) = delete;
    // CudaMemoryPool &operator=(CudaMemoryPool &other) = delete;

private:
    uint32_t create(std::list<MemUnit> &elementList, size_t elementSize, uint32_t elementCount)
    {
        uint32_t successCount = 0; 
        for (size_t i = 0; i < elementCount; i++)
        {
            void *element = 0;
            cudaError_t err = cudaMalloc((void **)&element, elementSize);
            if (err == cudaSuccess)
            {
                if (element != 0)
                {
                    elementList.push_back(MemUnit(element, false));
                    successCount++;
                }
            }
            else if(i == 0){
                throw std::runtime_error(cudaGetErrorString(err));
            }
            else{
                break;
            }
        }
        return successCount;
    }

private:
    std::unordered_map<size_t, std::unordered_map<size_t, std::list<MemUnit>>> m_elementListMap;
    std::mutex m_mtx; 
};

auto &cuda_mem_pool = CudaMemoryPool::instance();

bool mem_pool_alloc(size_t elementSize, void*&element, int device, EventState& event){
    return cuda_mem_pool.alloc(elementSize, element, device, event);
}

bool mem_pool_alloc(size_t elementSize, void*&element, int device, int stream_id){
    EventState old_event;
    return cuda_mem_pool.alloc(elementSize, element, device, old_event, stream_id);
}