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

namespace py = pybind11;

typedef std::tuple<py::array_t<int>, std::vector<int>, std::vector<int>, std::vector<std::string>, int> transformer_ret;

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
    void getChilds(int* idxs_ptr, std::vector<int>& f_arity, int& idxs_size, std::vector<std::vector<int>>& node_childs, int& func_len){
        
        std::vector<std::vector<int>> cur_arity_tmp;
        cur_arity_tmp.reserve(idxs_size * 2);
        cur_arity_tmp.push_back({0, f_arity[idxs_ptr[0]]});
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
            if(idxs_ptr[i] >= 0 && idxs_ptr[i] < func_len){
                cur_arity_tmp.push_back({i, f_arity[idxs_ptr[i]]});
            }
        }
    }
    
    std::mutex mtx;

    template<typename scalar_t>
    void transformer(const std::tuple<std::vector<std::string>, std::vector<int>, int>& f_attrs, 
                     const std::vector<std::vector<int>>& ind_after_cashes, const std::vector<std::vector<size_t>>& idxs, size_t sym_set_ptr,
                     std::vector<std::vector<int>>* cash_list, const std::vector<std::vector<int>>& records, const std::vector<std::string>& constants,
                     const std::vector<int>& paras, int* id_allocator, const int& cur_ind, const int& cur_posi, const int& compute_unit, std::vector<std::vector<std::vector<int>>>* exp_set,
                     std::vector<int>* record_posi, std::vector<std::string>* record_strs){
        
        int arguments_num = paras[0], exec_len_max = paras[2], pset_funcs_num = paras[3];
        std::unordered_map<std::string, std::array<int, 2>> output;
        int node_size = 0;
        for(int i = cur_ind; i < compute_unit + cur_ind; ++i){
            node_size += idxs[i][1];
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
        
        std::vector<std::string> f_name = std::get<0>(f_attrs);
        std::vector<int> f_arity = std::get<1>(f_attrs);
        int func_len = std::get<2>(f_attrs);
        int cur_expset_size = 0;
        int init_origin_posi = cur_posi;
        register int max_layer, child_size, idx, layer;
        std::string sym;
        std::vector<int> exps(exec_len_max);
        size_t ind_cashes_size;
        std::string sym_child;
        for(int k = cur_ind; k < idxs_size; ++k){
            int* idxs_ptr = (int*)(idxs[k][0]), idxs_ksize = idxs[k][1];
            std::vector<std::vector<int>> node_childs(idxs_ksize);
            
            if((*cash_list).size() > 0){
                for(int i = 0; i < (*cash_list)[k].size(); ++i){
                    mtx.lock();
                    output[sym_set[(*cash_list)[k][i]]] = {(*id_allocator), 0};
                    (*id_allocator) += 1;
                    mtx.unlock();
                }
            }
            getChilds(idxs_ptr, f_arity, idxs_ksize, node_childs, func_len);
            ind_cashes_size = ind_after_cashes[k].size();
            if(ind_after_cashes[k].size() == 1 && ind_after_cashes[k][0] != 0){
                ind_cashes_size = ind_after_cashes[k][0];
            }
            for(int ii = ind_cashes_size - 1; ii>=0; --ii){
                int i = ii, iter_i = i + init_origin_posi;
                idx = idxs_ptr[i];
                child_size = node_childs[i].size();
                if(idx >= 0 && idx < func_len){
                    assert (child_size == f_arity[idx]);
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
                    if(idx >= 0){
                        node_str = f_name[idx];
                    }
                    else{
                        node_str = constants[-idx - 1];
                    }
                    if(output.count(node_str) == 0){
                        if (idx >= 0){
                            output[node_str] = {idx - pset_funcs_num, 0};
                        }
                        else{
                            output[node_str] = {idx, 0};
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
                exps[0] = -1;
                exps[1] = 1;
                if (idx >= 0){
                    exps[2] = idx - pset_funcs_num;
                }
                else{
                    exps[2] = idx;
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
            for(int i = 0; i < records[k].size(); ++i){
                std::string sym = sym_set[records[k][i]];
                (*record_posi).push_back(output[sym][0]);
                (*record_strs).push_back(sym);
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
    m.def("tree2graph", [](std::tuple<std::vector<std::string>, std::vector<int>, int> f_attrs, 
                     std::vector<std::vector<int>> ind_after_cashes, std::vector<py::array_t<int>> idxs, size_t sym_set_ptr,
                     std::vector<std::vector<int>> cash_list, std::vector<std::vector<int>> records, std::vector<std::string> constants,
                     std::vector<int> paras){
        // printf("idxs: %d\n", idxs.size());
        long max_thread_num = 8;//sysconf(_SC_NPROCESSORS_ONLN) / 10;
        if (idxs.size() < max_thread_num){
            max_thread_num = 1;
        }
        int ind_num = idxs.size(), compute_unit = ceil(float(ind_num) / max_thread_num);
        int batch = ceil(ind_num / compute_unit), cur_posi = 0, cur_ind = 0;
        if (max_thread_num > batch){
            max_thread_num = batch;
        }
        
        std::vector<std::vector<size_t>> idxs_buf(ind_num);
        for(int k = 0; k < ind_num; ++k){
            idxs_buf[k] = {size_t(idxs[k].request().ptr), idxs[k].request().shape[0]};
        }

        std::thread* t_list = new std::thread[max_thread_num];
        clock_t st = std::clock();
        // std::vector<std::vector<std::vector<int>>> exp_set_final;
        std::vector<int> record_posi_final;
        std::vector<std::string> record_strs_final;
        std::vector<std::vector<std::vector<int>>> exp_set[batch];
        std::vector<std::string> record_strs[batch];
        std::vector<int> record_posi[batch];
        int id_allocator = paras[1];
        for(int k = 0; k < batch; ++k){
            // printf("Batch: %d, %d\n", k, batch);
            if(k == batch - 1){
                compute_unit = ind_num - k * compute_unit;
            }
            if(t_list[k % max_thread_num].joinable()){
                t_list[k % max_thread_num].join();
            }
            t_list[k % max_thread_num] = std::thread(transformer<scalar_t>, f_attrs, ind_after_cashes, idxs_buf, sym_set_ptr, &cash_list, records, constants, paras, &id_allocator, cur_ind, cur_posi, compute_unit, &(exp_set[k]), &(record_posi[k]), &(record_strs[k]));
            // transformer(f_attrs, ind_after_cashes, idxs, sym_set_ptr, cash_list, records, constants, paras, id_allocator, cur_ind, cur_posi, compute_unit, exp_set[k]);
            if(k < batch - 1){
                for(int i = 0; i < compute_unit; ++i){
                    cur_posi += idxs_buf[k * compute_unit + i][1];
                }
            }
            cur_ind += compute_unit;
        }
        for(int k = 0; k < max_thread_num; ++k){
            if(t_list[k].joinable()){
                t_list[k].join();
            }
        }
        // clock_t et = std::clock();
        // printf("t2g time 00 et - st: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
        std::vector<int> layer_info_final;
        int exec_len[batch] = {0}, exec_final_len = 0;
        for(int k = 0; k < batch; ++k){
            if(t_list[k % max_thread_num].joinable()){
                t_list[k % max_thread_num].join();
            }
            t_list[k % max_thread_num] = std::thread(exec_sum, &exec_len[k], &exp_set[k]);
        }
        for(int k = 0; k < max_thread_num; ++k){
            if(t_list[k].joinable()){
                t_list[k].join();
            }
        }

        for(int k = 0; k < batch; ++k){
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
        delete[] t_list;
        // et = std::clock();
        // printf("t2g time et - st: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
    
        return transformer_ret(exp_set_final, layer_info_final, record_posi_final, record_strs_final, id_allocator);
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