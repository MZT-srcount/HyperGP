


#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

#include"ops_backend.h"
#include<assert.h>

// #include "npp.h"
// #include "nppi.h"
// #include "npps.h"

using namespace gpu;
using namespace pygp_tensor;
using namespace pygp_img;

namespace py = pybind11;


template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_IMGSPROC(py::module& m){
    
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


    // m.def("SIFT", [](){

    // });

    // m.def("uLBP", [](){

    // });

    // m.def("HOG", [](){});
    // m.def("Hist", [](){});
    // m.def("DIF", [](){});
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
    
    m.def("to_numpy", [](Array<scalar_t>& a, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset, bool async){
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
        if(!async){
            cudaStreamSynchronize(streams[a.device_id][a.stream_id]);
        }
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

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void TEMPLATE_BIND_BROADCAST_DIM_2(py::module& m){
    
    

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
    m.def("ewise_pow_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 4);
    });

    m.def("ewise_pdiv_dim", [](const Array<scalar_t>& a, const Array<sscalar_t>& b, Array<tscalar_t>&out, int pre_dim_a=0, int post_dim_a, int pre_dim_b, int post_dim_b, int offset_a, int offset_b){
        operator_dim<scalar_t, sscalar_t, tscalar_t>(a, b, out, pre_dim_a, post_dim_a, pre_dim_b, post_dim_b, offset_a, offset_b, 5);
    });
}

template<typename scalar_t>
void TEMPLATE_BIND_MATRIX_DIM_1(py::module& m){

    
    handle_create();
    
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

}
template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_BROADCAST_DIM_1(py::module& m){

    
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

    m.def("compact", Compact<scalar_t>);
    m.def("concatenate", Concatenate<scalar_t>);

    

}

template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_FUNCS_DIM1(py::module& m){
    
    
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
    m.def("ewise_logfe", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 16);
    });
    m.def("ewise_logf2", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 17);
    });
    m.def("ewise_logf10", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 18);
    });
    m.def("ewise_sqrtf", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 19);
    });
    m.def("ewise_reciprocal", [](const Array<scalar_t>& a, Array<sscalar_t>& out, int32_t offset_a){
        ewise_compute_1op<scalar_t, sscalar_t>(a, out, offset_a, 20);
    });

    m.def("transfer_series", [](Array<scalar_t>& out, Array<scalar_t>& in_handle, int out_offset, int in_offset, int unit_len){
        if (in_handle.device_id == out.device_id){
            ewise_async_1op(in_handle, out);
            cudaMemcpyAsync(out.ptr + out_offset, in_handle.ptr + in_offset, unit_len * sizeof(scalar_t), cudaMemcpyDeviceToDevice, streams[out.device_id][out.stream_id]);
        }
        else{
            if(out.stream_id == -1) out.stream_id = rand() % STREAM_NUM_NDARRAY;
            if(in_handle.stream_id != -1) cudaStreamWaitEvent(streams[in_handle.device_id][in_handle.stream_id], in_handle.event_sign, 0);
        
            cudaMemcpyPeerAsync(out.ptr + out_offset, in_handle.device_id, in_handle.ptr + in_offset, in_handle.device_id, sizeof(scalar_t) * unit_len);
        }
    });
    
    m.def("transfer", [](Array<scalar_t>& out, Array<scalar_t>& in_handle, const std::vector<int32_t> unit_idxs, const std::vector<int32_t> unit_sizes, int32_t unit_len, int in_offset, int out_offset, int type){
        Array<scalar_t> in(0, out.device_id);
        scalar_t* in_ptr = cpy_gpus(in_handle, out, in);
        int thread_num = INIT_THREAD_NUM;
        int unit_total = unit_idxs.size();
        if (unit_total < thread_num){
            thread_num = unit_total + 32 - unit_total % 32;
        }
        int block_num = (unit_total + thread_num - 1) / thread_num;
        int32_t* idxs_gpu, *sizes_gpu;
        mem_pool_alloc_async(sizeof(int32_t) * unit_idxs.size(), (void**)&idxs_gpu, out.device_id, out.stream_id);
        mem_pool_alloc_async(sizeof(int32_t) * unit_sizes.size(), (void**)&sizes_gpu, out.device_id, out.stream_id);
        cudaMemcpyAsync((void*)sizes_gpu, (int32_t*)(unit_sizes.data()), sizeof(int32_t) * unit_sizes.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        cudaMemcpyAsync((void*)idxs_gpu, (int32_t*)(unit_idxs.data()), sizeof(int32_t) * unit_idxs.size(), cudaMemcpyHostToDevice, streams[out.device_id][out.stream_id]);
        // printf("%d, %d, %d, %d, %d\n", block_num, thread_num, unit_idxs.size(), offset, unit_idxs[0]);
        if (type == 0){
            transfer_get<scalar_t><<<block_num, thread_num, unit_sizes.size() * sizeof(int32_t), streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, sizes_gpu, unit_len, in_offset, out_offset, unit_sizes.size());
        }
        else if(type == 1){
            transfer_set<scalar_t><<<block_num, thread_num, unit_sizes.size() * sizeof(int32_t), streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, sizes_gpu, unit_len, in_offset, out_offset, unit_sizes.size());
        }
        else if(type == 2){
            transfer_list<scalar_t><<<block_num, thread_num, 0, streams[out.device_id][out.stream_id]>>>(out.ptr, in_ptr, idxs_gpu, unit_len, in_offset, out_offset, unit_idxs.size());
        }
        else{
            throw std::runtime_error(std::string("Wrong transfer type, only allow for 0, 1, 2, but ") + std::to_string(type) + "is given.\n");
            exit(-1);
        }
        cudaEventRecord(out.event_sign, streams[out.device_id][out.stream_id]);
        mem_pool_free_async((void*)idxs_gpu, out.device_id, out.stream_id);
        mem_pool_free_async((void*)sizes_gpu, out.device_id, out.stream_id);
        
    });
    
    m.def("get_properties", &get_properties);


    m.def("ewise_setitem", EwiseSetitem<scalar_t>);
    m.def("scalar_setitem", ScalarSetitem<scalar_t>);
}

template<typename scalar_t, typename sscalar_t>
void TEMPLATE_BIND_JUDGE(py::module& m){
    
    
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
        ewise_compute_2op<scalar_t, scalar_t, bool>(a, b, out, offset_a, offset_b, 10);
        cudaStreamSynchronize(streams[out.device_id][out.stream_id]);
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
}

template<typename scalar_t, typename sscalar_t, typename tscalar_t>
void TEMPLATE_BIND_FUNCS(py::module& m){
    

    
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
    
    //     m.def("ewise_eq", [](const Array<scalar_t>& a, const Array<bool>& b, bool& sign)){
    //         scalar_compute_2op(a, b, , 4);
    //     }


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

void broadcast_ops(py::module& m){

    // register_combinations_3([&m](auto type1, auto type2, auto type3){
    //     TEMPLATE_BIND_BROADCAST_DIM_2<decltype(type1), decltype(type2), decltype(type3)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{}, SupportedTypes_1{});
    
    // register_combinations_2([&m](auto type1, auto type2){
    //     TEMPLATE_BIND_BROADCAST_DIM_1<decltype(type1), decltype(type2)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{})
    
    // std::unordered_map<int, cudaStream_t*> streams = get_streams();
    TEMPLATE_BIND_BROADCAST_DIM_2<bool, bool, bool>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<uint8_t, uint8_t, uint8_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<uint16_t, uint16_t, uint16_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<uint32_t, uint32_t, uint32_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<uint64_t, uint64_t, uint64_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<int8_t, int8_t, int8_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<int16_t, int16_t, int16_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<int32_t, int32_t, int32_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<int64_t, int64_t, int64_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<float, float, float>(m);
    TEMPLATE_BIND_BROADCAST_DIM_2<double, double, double>(m);

    // TEMPLATE_BIND_BROADCAST_DIM_2<bool, bool, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint8_t, uint8_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint16_t, uint16_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint32_t, uint32_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint64_t, uint64_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int8_t, int8_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int16_t, int16_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int32_t, int32_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int64_t, int64_t, float>(m);
    
    // TEMPLATE_BIND_BROADCAST_DIM_2<bool, bool, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint8_t, uint8_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint16_t, uint16_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint32_t, uint32_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<uint64_t, uint64_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int8_t, int8_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int16_t, int16_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int32_t, int32_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_2<int64_t, int64_t, double>(m);

    TEMPLATE_BIND_BROADCAST_DIM_1<bool, bool>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<uint8_t, uint8_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<uint16_t, uint16_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<uint32_t, uint32_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<uint64_t, uint64_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<int8_t, int8_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<int16_t, int16_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<int32_t, int32_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<int64_t, int64_t>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<float, float>(m);
    TEMPLATE_BIND_BROADCAST_DIM_1<double, double>(m);

    // TEMPLATE_BIND_BROADCAST_DIM_1<bool, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint8_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint16_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint32_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint64_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int8_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int16_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int32_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int64_t, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<float, float>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<double, float>(m);

    // TEMPLATE_BIND_BROADCAST_DIM_1<bool, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint8_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint16_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint32_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<uint64_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int8_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int16_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int32_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<int64_t, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<float, double>(m);
    // TEMPLATE_BIND_BROADCAST_DIM_1<double, double>(m);

}

void basic_tensor_ops(py::module& m){

    
    // streams_create(std::vector<int>(1, 0), STREAM_NUM_NDARRAY);
    // register_combinations_3([&m](auto type1, auto type2, auto type3){
    //     TEMPLATE_BIND_FUNCS<decltype(type1), decltype(type2), decltype(type3)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{}, SupportedTypes_1{});
    // register_combinations_2([&m](auto type1, auto type2){
    //     TEMPLATE_BIND_FUNCS_DIM1<decltype(type1), decltype(type2)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{});

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

    TEMPLATE_BIND_FUNCS<bool, bool, bool>(m);
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

    TEMPLATE_BIND_FUNCS_DIM1<bool, float>(m);
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

    TEMPLATE_BIND_FUNCS_DIM1<bool, double>(m);
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

}

void judge_ops(py::module& m){
    
    
    // register_combinations_2([&m](auto type1, auto type2){
    //     TEMPLATE_BIND_JUDGE<decltype(type1), decltype(type2)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{});

    TEMPLATE_BIND_JUDGE<uint8_t, uint8_t>(m);
    TEMPLATE_BIND_JUDGE<uint16_t, uint16_t>(m);
    TEMPLATE_BIND_JUDGE<uint32_t, uint32_t>(m);
    TEMPLATE_BIND_JUDGE<uint64_t, uint64_t>(m);
    TEMPLATE_BIND_JUDGE<int8_t, int8_t>(m);
    TEMPLATE_BIND_JUDGE<int16_t, int16_t>(m);
    TEMPLATE_BIND_JUDGE<int32_t, int32_t>(m);
    TEMPLATE_BIND_JUDGE<int64_t, int64_t>(m);
    TEMPLATE_BIND_JUDGE<float, float>(m);
    TEMPLATE_BIND_JUDGE<double, double>(m);

    TEMPLATE_BIND_JUDGE<bool, float>(m);
    TEMPLATE_BIND_JUDGE<uint8_t, float>(m);
    TEMPLATE_BIND_JUDGE<uint16_t, float>(m);
    TEMPLATE_BIND_JUDGE<uint32_t, float>(m);
    TEMPLATE_BIND_JUDGE<uint64_t, float>(m);
    TEMPLATE_BIND_JUDGE<int8_t, float>(m);
    TEMPLATE_BIND_JUDGE<int16_t, float>(m);
    TEMPLATE_BIND_JUDGE<int32_t, float>(m);
    TEMPLATE_BIND_JUDGE<int64_t, float>(m);
    TEMPLATE_BIND_JUDGE<float, float>(m);
    TEMPLATE_BIND_JUDGE<double, float>(m);

    TEMPLATE_BIND_JUDGE<bool, double>(m);
    TEMPLATE_BIND_JUDGE<uint8_t, double>(m);
    TEMPLATE_BIND_JUDGE<uint16_t, double>(m);
    TEMPLATE_BIND_JUDGE<uint32_t, double>(m);
    TEMPLATE_BIND_JUDGE<uint64_t, double>(m);
    TEMPLATE_BIND_JUDGE<int8_t, double>(m);
    TEMPLATE_BIND_JUDGE<int16_t, double>(m);
    TEMPLATE_BIND_JUDGE<int32_t, double>(m);
    TEMPLATE_BIND_JUDGE<int64_t, double>(m);
    TEMPLATE_BIND_JUDGE<float, double>(m);
    TEMPLATE_BIND_JUDGE<double, double>(m);
}

void nn_ops(py::module& m){
    
    
    // std::unordered_map<int, cudaStream_t*> streams = get_streams();
    // register_combinations_2([&m](auto type1, auto type2){
    //     TEMPLATE_BIND_IMGSPROC<decltype(type1), decltype(type2)>(m);
    // }, SupportedTypes_1{}, SupportedTypes_1{});

    
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

    TEMPLATE_BIND_IMGSPROC<bool, float>(m);
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

    TEMPLATE_BIND_IMGSPROC<bool, double>(m);
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
void device_info(py::module& m){
    
    // std::unordered_map<int, cudaStream_t*> streams = get_streams();
    // streams_create(std::vector<int>(1, 0), STREAM_NUM_NDARRAY);
    m.def("set_device", &streams_create);
    m.def("is_available", &is_available);
    m.def("cuda_mem_available", &cuda_mem_available);
    m.def("cuda_mem_total", &cuda_mem_total);
    m.def("cuda_mem_ratio", &cuda_mem_ratio);
    m.def("check", [](){
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    });

}

void matrix_ops_gpu(py::module& m){
    
    // std::unordered_map<int, cudaStream_t*> streams = get_streams();
    TEMPLATE_BIND_MATRIX_DIM_1<uint8_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<uint16_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<uint32_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<uint64_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<int8_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<int16_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<int32_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<int64_t>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<float>(m);
    TEMPLATE_BIND_MATRIX_DIM_1<double>(m);

}

void initialize_global_streams() {
    // std::cout << "initialize_global_streams " << std::endl;
    for(int k = 0; k < STREAM_NUM_NDARRAY; ++k){
        cudaStreamCreate(&streams[0][k]);
    }
}

GlobalStreams gstreams;


PYBIND11_MODULE(ndarray_cuda_backend, m){
    py::class_<GlobalStreams>(m, std::string("GlobalStreams").c_str());
    initialize_global_streams();
    gstreams.init(streams);
    m.def("get_gstreams", [](){ 
        return gstreams;
    });

    m.def("set_gstreams", [](const GlobalStreams& gstreams){
        streams = gstreams.streams;
    });
    
    // 初始化模块1
    py::module m1 = m.def_submodule("matrix_ops_gpu", "matrix ops gpu");
    matrix_ops_gpu(m1);

    // 初始化模块2
    py::module m2 = m.def_submodule("device_info", "device info");
    device_info(m2);
    
    // 初始化模块3
    py::module m3 = m.def_submodule("nn_ops", "nn ops");
    nn_ops(m3);
    
    // 初始化模块4
    py::module m4 = m.def_submodule("judge_ops", "judge ops");
    judge_ops(m4);
    
    // 初始化模块5
    py::module m5 = m.def_submodule("basic_tensor_ops", "basic tensor ops");
    basic_tensor_ops(m5);
    
    // 初始化模块6
    py::module m6 = m.def_submodule("broadcast_ops", "broadcast ops");
    broadcast_ops(m6);
}