#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <thread>
#include <mutex>
#include <unistd.h>
#include <numeric>
#include <list>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include "cuda.h"

namespace py = pybind11;

typedef std::tuple<py::array_t<int>, std::vector<int>, std::vector<int>, std::vector<std::string>, std::vector<float>, int> transformer_ret;

namespace pygp_utils{

    #define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
    void getChilds(std::vector<int> const& node_arity, std::vector<std::vector<int>>& node_childs){
        
        std::vector<std::vector<int>> cur_arity_tmp(node_arity.size() * 2);
        cur_arity_tmp.push_back({0, node_arity[0]});
        
        int node_arity_ksize = node_arity.size();
        for(int i = 1; i < node_arity_ksize; ++i){
            int idx = cur_arity_tmp.back()[0];
            cur_arity_tmp.back()[1] -= 1;
            node_childs[idx].push_back(i);
            if (cur_arity_tmp.back()[1] == 0){
                cur_arity_tmp.pop_back();
            }
            if(node_arity[i] > 0){
                cur_arity_tmp.push_back({i, node_arity[i]});
            }
        }
    }

    #define PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
    void getChilds(float* idxs_ptr, std::vector<int>& f_arity, int& idxs_size, std::vector<std::vector<int>>& node_childs, int& func_len){
        
        std::vector<std::vector<float>> cur_arity_tmp;
        cur_arity_tmp.reserve(idxs_size * 2);
        cur_arity_tmp.push_back({0, idxs_ptr[1]});
        register int cur_arity;
        for(int i = 1; i < idxs_size; ++i){
            if(cur_arity_tmp.size() > 0){
                cur_arity = cur_arity_tmp.back()[1];
                if(cur_arity > 0){
                    node_childs[cur_arity_tmp.back()[0]].push_back(i);
                    cur_arity -= 1;
                }
                if (cur_arity <= 0){
                    cur_arity_tmp.pop_back();
                }
                else{
                    cur_arity_tmp.back()[1] = cur_arity;
                }
            }
            if(idxs_ptr[i * 3 + 2] == 0){
                cur_arity_tmp.push_back({float(i), idxs_ptr[i * 3 + 1]});
            }
        }
    }
    
    std::mutex mtx, mtx_constval;

    template<typename scalar_t>
    void transformer(const std::tuple<std::vector<std::string>, std::vector<int>, int>* f_attrs, 
                     const std::vector<size_t>* ind_after_cashes, const std::vector<size_t>* idxs, size_t sym_set_ptr,
                     std::vector<std::vector<int>>* cash_list, const std::vector<std::vector<int>>& records, std::vector<float>* constants,
                     const std::vector<int>& paras, int* id_allocator, const std::tuple<int, int, int>& basic_info, std::vector<std::vector<std::vector<int>>>* exp_set,
                     std::vector<int>* record_posi, std::vector<std::string>* record_strs, int* const_idx){
        int cur_ind = std::get<0>(basic_info);
        int cur_posi = std::get<1>(basic_info);
        int compute_unit = std::get<2>(basic_info);
        int arguments_num = paras[0], exec_len_max = paras[2], pset_funcs_num = paras[3];
        std::unordered_map<std::string, std::array<int, 2>> output;
        int node_size = 0;
        for(int i = cur_ind; i < compute_unit + cur_ind; ++i){
            node_size += (*idxs)[i * 2 + 1];
        }

        (*exp_set).reserve(node_size * exec_len_max);
        register std::string* sym_set;
        register bool pre_symset = false;
        if (sym_set_ptr == 0){
            sym_set = new std::string[node_size];
        }
        else{
            sym_set = (std::string*)sym_set_ptr + cur_posi;
            pre_symset = true;
        }
        
        register size_t idxs_size = compute_unit + cur_ind;
        
        std::vector<std::string> f_name = std::get<0>((*f_attrs));
        std::vector<int> f_arity;// = std::get<1>((*f_attrs));
        int func_len = std::get<2>((*f_attrs));
        int cur_expset_size = 0;
        int init_origin_posi = cur_posi;
        register int max_layer, child_size, idx, layer;
        std::string sym;
        std::vector<int> exps(exec_len_max);
        size_t ind_cashes_size;
        std::string sym_child;
        for(int k = cur_ind; k < idxs_size; ++k){
            float* idxs_ptr = (float*)((*idxs)[k * 2]);
            int idxs_ksize = (*idxs)[k * 2 + 1];
            std::vector<std::vector<int>> node_childs(idxs_ksize);
            
            if((*cash_list).size() > 0){
                int cash_list_ksize = (*cash_list)[k].size();
                std::vector<int>& cash_list_k = (*cash_list)[k];
                for(int i = 0; i < cash_list_ksize; ++i){
                    mtx.lock();
                    output[sym_set[cash_list_k[i]]] = {(*id_allocator), 0};
                    (*id_allocator) += 1;
                    mtx.unlock();
                }
            }
            getChilds(idxs_ptr, f_arity, idxs_ksize, node_childs, func_len);
            
            ind_cashes_size = (*ind_after_cashes)[k * 2 + 1];
            // if(ind_cashes_size == 1 && ind_after_cashes[k][0] != 0){
            //     ind_cashes_size = ind_after_cashes[k][0];
            // }
            int* preorder_offset = (int*)((*ind_after_cashes)[k * 2]);
            for(int ii = ind_cashes_size - 1; ii>=0; --ii){
                int i = preorder_offset[ii], iter_i = i + init_origin_posi;
                idx = int(idxs_ptr[i * 3]);
                child_size = node_childs[i].size();
                if(idxs_ptr[i * 3 + 2] == 0){
                    // assert (child_size == idxs_ptr[i * 3 + 1]);
                    // sym.reserve(100);
                    if(pre_symset){
                        sym = sym_set[i];
                    }
                    else{
                        sym = f_name[idx] + '(';
                    }

                    max_layer = 0;
                    
                    exps[0] = idx;
                    exps[1] = child_size;
                    for (int j = 0; j < child_size; ++j){
                        sym_child = sym_set[node_childs[i][j]];
                        layer = output[sym_child][1];
                        exps[j + 2] = output[sym_child][0];
                        
                        if(!pre_symset){
                            sym += sym_child + ", ";
                        }
                        if (layer > max_layer){
                            max_layer = layer;
                        }
                    }
                    if (!pre_symset){
                        sym = sym.erase(sym.size() - 2, 2) + ')';
                        sym_set[i] = sym;
                    }

                    if(output.count(sym) == 0 || i == 0){
                        
                        if (i == 0){
                            exps[child_size + 2] = arguments_num + k;
                            output[sym] = {arguments_num + k, max_layer + 1};
                        }
                        else{
                            /// [ ] TODO: record_dict should be replaced by list struct
                            mtx.lock();
                            exps[child_size + 2] = (*id_allocator);
                            output[sym] = {(*id_allocator), max_layer + 1};
                            (*id_allocator) += 1;
                            mtx.unlock();
                        }
                        
                        if (max_layer >= cur_expset_size){
                            (*exp_set).push_back({exps});
                            cur_expset_size += 1;
                        }
                        else{
                            (*exp_set)[max_layer].push_back(exps);
                        }
                    }

                }
                else{
                    /// [ ] TODO: unable to handle the self-define function.
                    
                    std::string node_str;
                    if(idxs_ptr[i * 3 + 2] == 1){
                        node_str = f_name[idx + pset_funcs_num];
                    }
                    else{
                        node_str = std::to_string(idxs_ptr[i * 3]);
                    }
                    if(output.count(node_str) == 0){
                        if(idxs_ptr[i * 3 + 2] == 1){
                            output[node_str] = {idx, 0};
                        }
                        else{
                            (*constants).push_back(idxs_ptr[i * 3]);
                            mtx_constval.lock();
                            output[node_str] = {-*const_idx - 1, 0};
                            *const_idx += 1;
                            mtx_constval.unlock();
                        }
                    }
                    if(!pre_symset){
                        sym_set[i] = node_str;
                    }
                }
            }
            // printf("ind_cashes_size: %d\n", ind_cashes_size);
            if(ind_cashes_size == 1){
                max_layer = 0;
                idx = idxs_ptr[0];
                exps[0] = pset_funcs_num - 1;
                exps[1] = 1;
                if(idxs_ptr[2] == 1){
                    exps[2] = idx;
                }
                else{
                    (*constants).push_back(idxs_ptr[0]);
                    mtx_constval.lock();
                    exps[2] = -*const_idx - 1;
                    *const_idx += 1;
                    mtx_constval.unlock();
                }
                exps[3] = arguments_num + k;
                
                if (0 >= cur_expset_size){
                    (*exp_set).push_back({exps});
                    cur_expset_size += 1;
                }
                else{
                    (*exp_set)[0].push_back(exps);
                }
            }
            int records_k_size = records[k].size();
            for(int i = 0; i < records_k_size; ++i){
                    
                std::string sym = sym_set[records[k][i]];
                (*record_posi).push_back(output[sym][0]);
                (*record_strs).push_back(sym);
                // printf("Here....%d, %d, %d, %d, %s\n", records[k][i], output[sym][0], cur_ind, compute_unit + cur_ind, sym.c_str());
            }
            init_origin_posi += idxs_ksize;
        }
        
        if(!pre_symset){
            delete[] sym_set;
        }
        // return transformer_ret(record_posi, record_strs, id_allocator);
    }

    void exec_sum(int* exec_len, std::vector<std::vector<std::vector<int>>>* exp_set){
        int exp_size1 = (*exp_set).size();
        for(int i = 0; i < exp_size1; ++i){
            *exec_len += (*exp_set)[i].size();
        }
    }
    void exec_cpy(size_t buf_ptr, std::vector<std::vector<int>>* exp_set){
        int* exp_final_set = (int*)buf_ptr;
        int exp_size2 = (*exp_set).size();
        for(int j = 0; j < exp_size2; ++j){
            int exp_size3 = (*exp_set)[j].size();
            for(int k = 0; k < exp_size3; ++k){
                exp_final_set[j * exp_size3 + k] = (*exp_set)[j][k];
            }
        }
    }

    __global__ void get_slice_list(float* encode_ptrs_gpu, int* preorder_ptrs_gpu, int* encode_posis_gpu, int* slice_gpu, int total_len, int* func_flags, int* const_flags){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int t_n = blockDim.x * gridDim.x;
        int posi_index, idx, total, pre_posi;
        for(int i = tid; i < total_len; i += t_n){
            idx = i;
            pre_posi = encode_posis_gpu[idx];
            posi_index = preorder_ptrs_gpu[i] + pre_posi;
            total = encode_ptrs_gpu[posi_index * 3 + 1];
            if(total > 0){
                func_flags[i] = 1;
                while(total > 0 && idx < total_len){
                    idx += 1;
                    posi_index = preorder_ptrs_gpu[idx] + pre_posi;
                    total += encode_ptrs_gpu[posi_index * 3 + 1] - 1;
                }
            }
            else if(encode_ptrs_gpu[posi_index * 3 + 2] == 2){
                const_flags[i] = 1;
            }
            slice_gpu[i] = idx;
        }
    }

     __global__ void get_expr_list(float* encode_ptrs_gpu, int* preorder_ptrs_gpu, int* encode_posis_gpu, int* slice_gpu, int total_len, int* pre_sums, int* pre_sums_consts, int max_arity, int* exprs_gpu, float* consts_gpu, int init_id_allocator){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int t_n = blockDim.x * gridDim.x;
        int posi_index, idx, total, pre_posi, child_type;
        int unit_len = max_arity + 3;
        int total_exprs = pre_sums[total_len - 1], total_consts = pre_sums_consts[total_len - 1];
        for(int i = total_len - 1; i >= 0; i -= t_n){
            pre_posi = encode_posis_gpu[i];
            posi_index = preorder_ptrs_gpu[i] + pre_posi;
            if(encode_ptrs_gpu[posi_index * 3 + 1] > 0){
                exprs_gpu[(total_exprs - pre_sums[i]) * unit_len] = encode_ptrs_gpu[posi_index * 3];//func
                exprs_gpu[(total_exprs - pre_sums[i]) * unit_len + 1] = encode_ptrs_gpu[posi_index * 3 + 1];//arity
                int offset = i + 1;
                for(int j = 0; j < encode_ptrs_gpu[posi_index * 3 + 1]; ++j){
                    child_type = encode_ptrs_gpu[(preorder_ptrs_gpu[offset] + pre_posi) * 3 + 2];
                    if(child_type == 0){
                        exprs_gpu[(total_exprs - pre_sums[i]) * unit_len + 2 + j] = (total_exprs - pre_sums[offset]) + init_id_allocator;//childs
                    }
                    else if(child_type == 1){
                        exprs_gpu[(total_exprs - pre_sums[i]) * unit_len + 2 + j] = encode_ptrs_gpu[(preorder_ptrs_gpu[offset] + pre_posi) * 3];//childs
                    }
                    else{
                        exprs_gpu[(total_exprs - pre_sums[i]) * unit_len + 2 + j] = -pre_sums_consts[offset];//childs
                        consts_gpu[(total_consts - pre_sums_consts[offset]) - 1] = encode_ptrs_gpu[(preorder_ptrs_gpu[offset] + pre_posi) * 3];
                    }
                    offset = slice_gpu[offset] + 1;
                }
                exprs_gpu[(total_exprs - pre_sums[i]) * unit_len + unit_len - 1] = init_id_allocator + (total_exprs - pre_sums[i]);//arity
            }
        }
    }
}

template<typename scalar_t> 
void TEMPLATE_BIND_FUNCS(py::module& m){
    
    #include <ctime>
    using namespace pygp_utils;
    m.def("test", [](const std::vector<std::vector<py::object>>& res){
        std::vector<std::vector<int>> idxs;
        for(int i = 0; i < res.size(); ++i){
            std::vector<int> idx;
            idx.reserve(res[i].size());
            for(int j = 0; j < res[i].size(); ++j){
                const py::int_& arity = res[i][j].attr("arity"), idx_int = res[i][j].attr("idx");
                if(idx_int != -1){
                    if (arity > 0){
                        idx.push_back(arity + 10);
                    }
                    else{
                        idx.push_back(arity);
                    }
                }
                else{
                    int a;
                }
                // idxs.push_back(res[i][j].attr("arity").cast<int>());
            }
            idxs.push_back(idx);
        }
        printf("here,,,succeed!!!%d\n", res[0][0].attr("arity").cast<int>());
    });
    m.def("tree2graph_v2", [](std::vector<py::array_t<float>> encode_arrays, std::vector<py::array_t<int>> preorder_idxs, std::vector<std::vector<int>> records, std::vector<std::vector<int>> cash_list, int init_id_allocator, int max_arity){
        int* encode_sizes = new int[encode_arrays.size()];
        float* encode_ptrs_gpu;
        int* preorder_ptrs_gpu;
        int* encode_sizes_gpu;
        int* slice_gpu;
        float* consts_gpu;
        int* exprs_gpu;
        // printf("000900\n");
        // cudaDeviceSynchronize();
        // cudaError_t err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        cudaStream_t stream_tmp;
        cudaStreamCreate(&stream_tmp);

        int total_len = 0;
        for(int i = 0; i < encode_arrays.size(); ++i){
            encode_sizes[i] = total_len;
            total_len += encode_arrays[i].request().shape[0] / 3;
        }
        
        thrust::device_vector<int> func_flags(total_len, 0);
        thrust::device_vector<int> prefix_sum_funcs(total_len);
        
        thrust::device_vector<int> const_flags(total_len, 0);
        thrust::device_vector<int> prefix_sum_consts(total_len);
        // printf("000800\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        cudaMallocAsync((void**)&slice_gpu, total_len * sizeof(float), stream_tmp); 
        cudaMallocAsync((void**)&encode_ptrs_gpu, total_len * 3 * sizeof(float), stream_tmp); 
        cudaMallocAsync((void**)&preorder_ptrs_gpu, total_len * sizeof(float), stream_tmp); 
        int* encode_posis = new int[total_len], *encode_posis_gpu;
        int end_posi;
        // printf("000700\n");
        // cudaDeviceSynchronize();
        //  err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        cudaMallocAsync((void**)&encode_posis_gpu, total_len * sizeof(int), stream_tmp); 
        for(int i = 0; i < encode_arrays.size(); ++i){
            cudaMemcpyAsync((encode_ptrs_gpu + encode_sizes[i] * 3), encode_arrays[i].request().ptr, encode_arrays[i].request().shape[0] * sizeof(float), cudaMemcpyHostToDevice, stream_tmp);
            cudaMemcpyAsync((preorder_ptrs_gpu + encode_sizes[i]), preorder_idxs[i].request().ptr, preorder_idxs[i].request().shape[0] * sizeof(int), cudaMemcpyHostToDevice, stream_tmp);
            end_posi = (i < (encode_arrays.size() - 1)) ? encode_sizes[i + 1] : total_len;
            for(int j = encode_sizes[i]; j < end_posi; ++j){
                encode_posis[j] = encode_sizes[i];
            }
        }
        cudaMemcpyAsync(encode_posis_gpu, encode_posis, total_len * sizeof(int), cudaMemcpyHostToDevice, stream_tmp);
        // printf("000700\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        int thread_num = total_len < 256 ? int(total_len / 32 + 1) * 32 : 256;
        int batch = ceil(total_len / (thread_num * 10));

        get_slice_list<<<batch, thread_num>>>(encode_ptrs_gpu, preorder_ptrs_gpu, encode_posis_gpu, slice_gpu, total_len, thrust::raw_pointer_cast(func_flags.data()), thrust::raw_pointer_cast(const_flags.data()));
        // printf("00000\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        // printf("000100\n");
        thrust::exclusive_scan(func_flags.begin(), func_flags.end(), prefix_sum_funcs.begin());
        thrust::exclusive_scan(const_flags.begin(), const_flags.end(), prefix_sum_consts.begin());
        
        
        // printf("000200\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));

        cudaMallocAsync((void**)&consts_gpu, (const_flags[const_flags.size() - 1] + 1) * sizeof(float), stream_tmp); 
        cudaMallocAsync((void**)&exprs_gpu, (func_flags[func_flags.size() - 1] + 1) * sizeof(int) * (max_arity + 3), stream_tmp); 
        // printf("000200\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));
        get_expr_list<<<batch, thread_num>>>(encode_ptrs_gpu, preorder_ptrs_gpu, encode_posis_gpu, slice_gpu, total_len, thrust::raw_pointer_cast(prefix_sum_funcs.data()), thrust::raw_pointer_cast(prefix_sum_consts.data()), max_arity, exprs_gpu, consts_gpu, init_id_allocator);
        
        // printf("000300\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));
        py::array_t<int> exprs(func_flags[func_flags.size() - 1] * (max_arity + 3));
        cudaMemcpyAsync(exprs.request().ptr, exprs_gpu, func_flags[func_flags.size() - 1] * (max_arity + 3) * sizeof(int), cudaMemcpyDeviceToHost, stream_tmp);
        py::array_t<float> consts(const_flags[const_flags.size() - 1]);
        cudaMemcpyAsync(consts.request().ptr, consts_gpu, const_flags[const_flags.size() - 1] * sizeof(float), cudaMemcpyDeviceToHost, stream_tmp);
        
        delete[] encode_posis;
        delete[] encode_sizes;
        cudaStreamDestroy(stream_tmp);

        // printf("000400\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));
        cudaFree(slice_gpu);
        cudaFree(encode_ptrs_gpu);
        cudaFree(preorder_ptrs_gpu);
        cudaFree(consts_gpu);
        cudaFree(exprs_gpu);
        
        // printf("000500\n");
        // cudaDeviceSynchronize();
        // err_l = cudaGetLastError();
        // if (err_l != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err_l));
        return std::tuple<py::array_t<int>, py::array_t<float>>(exprs, consts);
    });
    m.def("tree2graph", [](std::tuple<std::vector<std::string>, std::vector<int>, int> f_attrs, 
                     pybind11::list& ind_after_cashes, pybind11::list& idxs, size_t sym_set_ptr,
                     std::vector<std::vector<int>>& cash_list, std::vector<std::vector<int>>& records, std::vector<int>& paras){
        // printf("idxs: %d\n", idxs.size());
        long max_thread_num = 10;//sysconf(_SC_NPROCESSORS_ONLN) / 10;
        int idxs_len = py::len(idxs);
        if (idxs_len < max_thread_num){
            max_thread_num = 1;
        }
        int ind_num = idxs_len, compute_unit = ceil(float(ind_num) / max_thread_num);
        int batch = ceil(ind_num / compute_unit), cur_posi = 0, cur_ind = 0;
        if (max_thread_num > batch){
            max_thread_num = batch;
        }
        
        std::vector<size_t> idxs_buf(ind_num * 2);
        std::vector<size_t> buf_ind_after_cashes(ind_num * 2);
        py::buffer_info idxs_buf_tmp, buf_ind_after_cashes_tmp;
        for(int k = 0; k < ind_num; ++k){
            idxs_buf_tmp = py::reinterpret_borrow<py::array_t<float>>(idxs[k]).request();
            buf_ind_after_cashes_tmp = py::reinterpret_borrow<py::array_t<int>>(ind_after_cashes[k]).request();
            idxs_buf[k * 2] = size_t(idxs_buf_tmp.ptr), idxs_buf[k * 2 + 1] = size_t(idxs_buf_tmp.shape[0] / 3);
            buf_ind_after_cashes[k * 2] = size_t(buf_ind_after_cashes_tmp.ptr), buf_ind_after_cashes[k * 2 + 1] = size_t(buf_ind_after_cashes_tmp.shape[0]);
        }

        std::thread* t_list = new std::thread[max_thread_num];
        // clock_t st = std::clock();
        // std::vector<std::vector<std::vector<int>>> exp_set_final;
        std::vector<int> record_posi_final;
        std::vector<std::string> record_strs_final;
        std::vector<std::vector<std::vector<int>>> exp_set[batch];
        std::vector<std::string> record_strs[batch];
        std::vector<int> record_posi[batch];
        int id_allocator = paras[1];
        int const_idx = 0;
        std::vector<float> constants;
        for(int k = 0; k < batch - 1; ++k){
            // printf("Batch: %d, %d\n", k, batch);
            // if(k == batch - 1){
            //     compute_unit = ind_num - k * compute_unit;
            // }
            if(t_list[k % max_thread_num].joinable()){
                t_list[k % max_thread_num].join();
            }
            std::tuple<int, int, int> basic_info = std::tuple<int, int, int>(cur_ind, cur_posi, compute_unit);
            t_list[k % max_thread_num] = std::thread(transformer<scalar_t>, &f_attrs, &buf_ind_after_cashes, &idxs_buf, sym_set_ptr, &cash_list, records, &constants, paras, &id_allocator, basic_info, &(exp_set[k]), &(record_posi[k]), &(record_strs[k]), &const_idx);
            // transformer(f_attrs, ind_after_cashes, idxs, sym_set_ptr, cash_list, records, constants, paras, id_allocator, cur_ind, cur_posi, compute_unit, exp_set[k]);
            // if(k < batch - 1){
                for(int i = 0; i < compute_unit; ++i){
                    cur_posi += idxs_buf[(k * compute_unit + i) * 2 + 1];
                }
            // }
            cur_ind += compute_unit;
        }
        compute_unit = ind_num - (batch - 1) * compute_unit;
        std::tuple<int, int, int> basic_info = std::tuple<int, int, int>(cur_ind, cur_posi, compute_unit);
        transformer<scalar_t>(&f_attrs, &buf_ind_after_cashes, &idxs_buf, sym_set_ptr, &cash_list, records, &constants, paras, &id_allocator, basic_info, &(exp_set[batch - 1]), &(record_posi[batch - 1]), &(record_strs[batch - 1]), &const_idx);
        // for(int k = 0; k < max_thread_num; ++k){
        //     if(t_list[k].joinable()){
        //         t_list[k].join();
        //     }
        // }
        // clock_t et = std::clock();
        // printf("t2g time 00 et - st: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
        std::vector<int> layer_info_final;
        layer_info_final.reserve(idxs_buf[1]);
        int exec_len[batch] = {0}, exec_final_len = 0;
        for(int k = 0; k < batch - 1; ++k){
            if(t_list[k % max_thread_num].joinable()){
                t_list[k % max_thread_num].join();
            }
            t_list[k % max_thread_num] = std::thread(exec_sum, &exec_len[k], &exp_set[k]);
        }
        exec_sum(&exec_len[batch - 1], &exp_set[batch - 1]);

        for(int k = 0; k < batch; ++k){
            if(t_list[k % max_thread_num].joinable()){
                t_list[k % max_thread_num].join();
            }
            exec_final_len += exec_len[k];
            for(int z = 0; z < exp_set[k].size(); ++z){
                if(z >= layer_info_final.size()){
                    layer_info_final.push_back(exp_set[k][z].size());
                }
                else{
                    layer_info_final[z] += exp_set[k][z].size();
                }
            }
        }
        exec_final_len *= paras[2];
        py::array_t<int> exp_set_final(exec_final_len);
        size_t exp_set_ptr = size_t(exp_set_final.request().ptr);
        std::vector<size_t> init_posi, accumulate_posi;
        if (layer_info_final.size() > 0){
            init_posi.resize(layer_info_final.size());
            accumulate_posi.resize(layer_info_final.size());
            init_posi[0] = 0;
        }
        for(int i = 1; i < layer_info_final.size(); ++i){
            init_posi[i] = layer_info_final[i - 1] + init_posi[i - 1];
            accumulate_posi[i] = 0;
        }
        for(int k = 0; k < batch; ++k){
            for(int i = 0; i < exp_set[k].size(); ++i){
                exec_cpy(size_t(exp_set_ptr + (init_posi[i] + accumulate_posi[i]) * paras[2] * sizeof(int)), &exp_set[k][i]);
                accumulate_posi[i] += exp_set[k][i].size();
            }
            for(int i = 0; i < record_posi[k].size(); ++i){
                record_posi_final.push_back(record_posi[k][i]);
                record_strs_final.push_back(record_strs[k][i]);
            }
        }
        if(sym_set_ptr == 0){
            delete[] (std::string*)sym_set_ptr;
        }
        
        for(int k = 0; k < max_thread_num; ++k){
            if(t_list[k].joinable()){
                t_list[k].join();
            }
        }
        delete[] t_list;
        // et = std::clock();
        // printf("t2g time et - st: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    
        return transformer_ret(exp_set_final, layer_info_final, record_posi_final, record_strs_final, constants, id_allocator);
    });
}
PYBIND11_MODULE(pygp_utils, m){
    
    TEMPLATE_BIND_FUNCS<int8_t>(m);
    TEMPLATE_BIND_FUNCS<int32_t>(m);
    TEMPLATE_BIND_FUNCS<int64_t>(m);
    TEMPLATE_BIND_FUNCS<float>(m);
    TEMPLATE_BIND_FUNCS<double>(m);
    // m.def("tree2graph", &transformer);
}