#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include <pybind11/numpy.h>
#include<iostream>
#include<vector>
#include<unordered_map>
#include<string>
#include<list>
#include <algorithm>

namespace py = pybind11;
typedef double scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
// typedef std::tuple<std::string, scalar_t*, std::vector<size_t>, std::vector<size_t>> list_elem;
typedef std::tuple<std::string, py::array_t<scalar_t>> list_elem;
typedef std::tuple<std::vector<py::array_t<scalar_t>>, std::vector<size_t>,
     std::vector<std::string>, std::vector<std::size_t>> cash_ret;
typedef std::tuple<std::vector<py::array_t<scalar_t>>,
     std::vector<std::vector<std::size_t>>, size_t, std::vector<std::vector<std::size_t>>> cashes_ret;

namespace pygp_cash{

    void getChilds(std::vector<size_t> const& node_arity, std::vector<std::vector<size_t>>& node_childs, size_t init_origin_posi){
        
        size_t node_arity_ksize = node_arity.size();
        std::vector<std::vector<size_t>> cur_arity_tmp = {{init_origin_posi, node_arity[0]}};
        for(int i = 1; i < node_arity_ksize; ++i){
            size_t idx = cur_arity_tmp.back()[0];
            cur_arity_tmp.back()[1] -= 1;
            node_childs[idx].push_back(i);
            if (cur_arity_tmp.back()[1] == 0){
                cur_arity_tmp.pop_back();
            }
            if(node_arity[i] > 0){
                cur_arity_tmp.push_back({i + init_origin_posi, node_arity[i]});
            }
        }

    }

    size_t get_size(std::vector<size_t> const & shape){
        size_t _size = 1;
        size_t shape_size = shape.size();
        for(int i = 0; i < shape_size; ++i){
            _size *= shape[i];
        }
        return _size;
    }

    std::vector<size_t> get_strides(std::vector<size_t> const & shape){
        size_t _stride = 1;
        std::vector<size_t> _strides = {_stride};
        for(int i = shape.size() - 1; i > 0; --i){
            _stride *= shape[i];
            _strides.push_back(_stride);
        }
        std::reverse(_strides.begin(), _strides.end());
        return _strides;
    }

    struct CashList{
        
        CashList(const size_t& l_size){
            this->limited_size = l_size;
        }
        ~CashList(){}
        bool insert(std::string key, py::array_t<scalar_t> a, std::vector<size_t> const& shape){
            // size_t size = get_size(shape);
            // std::vector<size_t> strides = get_strides(shape);
            // scalar_t* c = (scalar_t*)std::malloc(ELEM_SIZE * size);
            // std::memcpy(c, a.request().ptr, ELEM_SIZE * size);
            // _cash_list.push_front(list_elem(key, c, shape, strides));
            _cash_list.push_front(list_elem(key, a));
            cashes[key] = _cash_list.begin();
            if(this->limited_size < _cash_list.size()){
                list_elem l_tmp = _cash_list.back();
                this->_delete(std::get<0>(l_tmp));
            }
            return true;
        }

        bool _delete(std::string const& key){
            if (cashes.find(key) == cashes.end()){
                return false;
            }
            // list_elem l_tmp = *cashes[key];
            // delete std::get<1>(l_tmp);
            _cash_list.erase(cashes[key]);
            
            cashes.erase(key);
            return true;
        }

        bool reinsert(std::string key, size_t const& posi){
            if (cashes.find(key) == cashes.end()){
                return false;
            }
            list_elem l_tmp = *cashes[key];
            _cash_list.erase(cashes[key]);
            auto iter = _cash_list.begin();
            std::advance(iter, posi);
            std::list<list_elem>::iterator new_iter = _cash_list.insert(iter, l_tmp);
            cashes[key] = new_iter;
            return true;
        }

        bool findElem(std::string key){
            if(cashes.find(key) == cashes.end()){
                return false;
            }
            return true;
        }

        py::array_t<scalar_t> get(std::string key){
            if(cashes.find(key) == cashes.end()){
                return py::array_t<scalar_t>(-1);
            }
            list_elem l_tmp = *cashes[key];
            // std::vector<size_t> numpy_strides = std::get<3>(l_tmp);
            // std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(), [](size_t& c){return c * ELEM_SIZE;});
            // return py::array_t<scalar_t>(std::get<2>(l_tmp), std::get<3>(l_tmp), std::get<1>(l_tmp));
            return std::get<1>(l_tmp);
        }

        std::list<list_elem> _cash_list;
        std::unordered_map<std::string, std::list<list_elem>::iterator> cashes;
        size_t limited_size;
    };

    
    cash_ret getCash(CashList& cashes_m, std::vector<size_t> const& node_arity,
            std::vector<std::string> const& node_str, std::vector<size_t> const& records){

        /// ret vars 
        std::vector<py::array_t<scalar_t>> cash_arrays;
        std::vector<size_t> cash_list;
        std::vector<size_t> ind_after_cash;
        std::vector<std::string> sym_set(node_arity.size());

        /// temp vars
        std::vector<size_t> cash_sign;
        bool cash_detect = false;
        size_t init_origin_posi = 0;
        
        std::vector<std::vector<size_t>> childs(node_arity.size());
        getChilds(node_arity, childs, init_origin_posi);
        /// scan the node list to search cash nodes
        for(int i = node_arity.size() - 1; i>=0; --i){
            if (node_arity[i] != 0){
                std::string sym = node_str[i] + '(';
                size_t child_size = childs[i].size();
                for(size_t j = 0; j < child_size; ++j){
                    sym += sym_set[childs[i][j]] + ", ";
                }
                sym = sym.substr(0, sym.size() - 2) + ')';
                sym_set[i] = sym;
                if (cashes_m.findElem(sym)){
                    cash_sign.push_back(i);
                    cash_detect = true;
                }
            }
            else{
                sym_set[i] = node_str[i];
            }
        }
        
        /// if successfully detect cash node
        if(cash_detect){
            size_t cash_num = cash_sign.size(), scan_iter = cash_sign.back(), last_ind = 0;
            for(int i = cash_num - 1; i >= 0; --i){
                if(cash_sign[i] < scan_iter){
                    continue;
                }
                size_t cur_posi = cash_sign[i], cur_arity = node_arity[i], init_posi = cash_sign[i];
                for(int j = last_ind; j < init_posi; ++j){
                    ind_after_cash.push_back(j);
                }
                last_ind = init_posi;
                bool suc = true;
                while(cur_arity > 0){
                    cur_arity += node_arity[cur_posi + 1] - 1;
                    cur_posi += 1;
                }
                for(int j = 0; j < records.size(); ++j){
                    if(init_posi < records[j] && records[j] <= cur_posi){
                        suc = false;
                        break;
                    }
                }
                if(suc){
                    size_t suc_id = cash_sign[i];
                    cash_list.push_back(suc_id);
                    cash_arrays.push_back(cashes_m.get(sym_set[suc_id]));
                    scan_iter = cur_posi + 1;
                    last_ind = scan_iter;
                }
            }
            for(int j = last_ind; j < node_arity.size(); ++j){
                ind_after_cash.push_back(j);
            }
        }
        else{
            for(int j = 0; j < node_arity.size(); ++j){
                ind_after_cash.push_back(j);
            }
        }
        return std::tuple<std::vector<py::array_t<scalar_t>>,
     std::vector<std::size_t>, std::vector<std::string>, std::vector<std::size_t>>(cash_arrays, cash_list, sym_set, ind_after_cash);
    }

    #include <ctime>
    cashes_ret getCashes(CashList& cashes_m, std::tuple<std::vector<std::string>, std::vector<size_t>, int> f_attrs,
                        std::vector<std::vector<int>> idxs, std::vector<scalar_t> const& constants, std::vector<std::vector<size_t>> const& records){
        
        clock_t st = std::clock();
        size_t idxs_size = idxs.size();
        /// ret vars 
        std::vector<py::array_t<scalar_t>> cash_arrays;
        std::vector<std::vector<size_t>> cash_list(idxs_size);
        std::vector<std::vector<size_t>> ind_after_cash(idxs_size);

        size_t node_arity_size = 0;
        for(int i = 0; i < idxs_size; ++i){
            node_arity_size += idxs[i].size();
        }
        std::string* sym_set = new std::string[node_arity_size];
        // std::vector<std::string> sym_set(node_arity_size);
    
        /// temp vars
        std::vector<std::string> f_name = std::get<0>(f_attrs);
        std::vector<size_t> f_arity = std::get<1>(f_attrs);
        int func_len = std::get<2>(f_attrs);
        std::unordered_map<std::string, bool> cash_already;
        bool cash_detect = false;
        size_t init_origin_posi = 0;
        for(int k = 0; k < idxs_size; ++k){
            
            std::vector<size_t> cash_sign;
            std::vector<std::vector<size_t>> childs(idxs[k].size());
            std::vector<size_t> node_arity;
            std::vector<std::string> node_str;
            
            for(int i = 0; i < idxs[k].size(); ++i){
                int idx = idxs[k][i];
                if(idx >= 0){
                    node_arity.push_back(f_arity[idx]);
                    node_str.push_back(f_name[idx]);
                }
                else{
                    node_arity.push_back(0);
                    node_str.push_back(std::to_string(constants[-idx]));
                }
            }
            getChilds(node_arity, childs, 0);
            /// scan the node list to search cash nodes
            for(int i = node_arity.size() - 1; i>=0; --i){
                if (node_arity[i] != 0){
                    std::string sym = node_str[i] + '(';
                    size_t child_size = childs[i].size();
                    for(size_t j = 0; j < child_size; ++j){
                        sym += sym_set[childs[i][j] + init_origin_posi] + ", ";
                    }
                    sym = sym.substr(0, sym.size() - 2) + ')';
                    sym_set[i + init_origin_posi] = sym;
                    if (cashes_m.findElem(sym)){
                        cash_sign.push_back(i);
                        cash_detect = true;
                    }
                }
                else{
                    sym_set[i + init_origin_posi] = node_str[i];
                }
            }
            /// if successfully detect cash node
            if(cash_detect){
                size_t cash_num = cash_sign.size(), scan_iter = cash_sign.back(), last_ind = 0;
                for(int i = cash_num - 1; i >= 0; --i){
                    size_t cash_idx = cash_sign[i];
                    if(cash_idx < scan_iter || cash_already.find(sym_set[cash_idx + init_origin_posi]) != cash_already.end()){
                        continue;
                    }
                    size_t cur_posi = cash_idx, cur_arity = node_arity[cash_idx], init_posi = cash_idx;
                    for(int j = last_ind; j < init_posi; ++j){
                        ind_after_cash[k].push_back(j);
                    }
                    last_ind = init_posi;
                    bool suc = true;
                    while(cur_arity > 0){
                        // printf("!!!777777766777777!!!!%d, %d, %d\n", cur_posi, node_arity.size(), node_arity[cur_posi]);
                        cur_arity += node_arity[cur_posi + 1] - 1;
                        cur_posi += 1;
                    }
                    for(int j = 0; j < records[k].size(); ++j){
                        if(init_posi < records[k][j] && records[k][j] <= cur_posi){
                            suc = false;
                            break;
                        }
                    }
                    if(suc){
                        size_t suc_id = cash_idx;
                        cash_list[k].push_back(suc_id);
                        cash_arrays.push_back(cashes_m.get(sym_set[suc_id + init_origin_posi]));
                        cash_already[sym_set[suc_id + init_origin_posi]] = true;
                        scan_iter = cur_posi + 1;
                        last_ind = scan_iter;
                    }
                }
                for(int j = last_ind; j < idxs[k].size(); ++j){
                    ind_after_cash[k].push_back(j);
                }
            }
            else{
                for(int j = 0; j < idxs[k].size(); ++j){
                    ind_after_cash[k].push_back(j);
                }
            }
            init_origin_posi += idxs[k].size();
        }
        clock_t et = std::clock();
        printf("time et - st: %f\n", (double)(et - st) / CLOCKS_PER_SEC);
        return cashes_ret(cash_arrays, cash_list, (size_t)sym_set, ind_after_cash);
    }
}

PYBIND11_MODULE(pygp_cash, m){
    namespace py = pybind11;
    using namespace pygp_cash;
    py::class_<CashList>(m, "CashList")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def("insert", &CashList::insert)
        .def("delete", &CashList::_delete)
        .def("reinsert", &CashList::reinsert)
        .def("findElem", &CashList::findElem)
        .def("get", &CashList::get);

    m.def("getCash", &getCash);
    m.def("getCashes", &getCashes);

}