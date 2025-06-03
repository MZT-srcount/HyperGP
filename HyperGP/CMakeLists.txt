cmake_minimum_required(VERSION 3.2)

project(PyGP_Tensor_backend CXX CUDA)

execute_process(COMMAND python3-config --prefix
    OUTPUT_VARIABLE Python_ROOT_DIR)

find_package(Python COMPONENTS Development Interpreter)
include_directories(${Python_INCLUDE_DIRS})

execute_process(COMMAND python3 -m pybind11 --cmakedir
    RESULT_VARIABLE __pybind_exit_code
    OUTPUT_VARIABLE __pybind_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(pybind11 PATHS ${__pybind_path})

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -march=x86-64 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
else()
    set(CMAKE_CXX_FLAGS "/std:c++11 -O3 -march=x86-64 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})

############################################################################
### CPU BACKEND
############################################################################

string(REGEX REPLACE "-rdynamic" "" CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}")
string(REGEX REPLACE "-rdynamic" "" CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS}")

### tree2graph_transformer

# add_library(pygp_utils MODULE src/utils_backend.cc)
# target_link_libraries(pygp_utils ${LINKER_LIBS})
# pybind11_extension(pygp_utils)
# pybind11_strip(pygp_utils)

# set_target_properties(pygp_utils
#     PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
#     CXX_VISIBILITY_PRESET "hidden"
#     LINK_FLAGS "-Wl,-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections"
# )

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET tree2graph_transformer PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

### cash_backend

# string(REGEX REPLACE "-rdynamic" "" CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}")
# string(REGEX REPLACE "-rdynamic" "" CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS}")

add_library(pygp_cash MODULE src/cash_backend.cc)
target_link_libraries(pygp_cash ${LINKER_LIBS})
pybind11_extension(pygp_cash)
pybind11_strip(pygp_cash)

set_target_properties(pygp_cash
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    CXX_VISIBILITY_PRESET "hidden"    
    LINK_FLAGS "-Wl,-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET pygp_cash PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



############################################################################
### CUDA BACKEND
############################################################################
# set(CUDA_TOOLKIT_ROOT_DIR /home/mazt/cuda-11.4.2)
set(CUDA_NVCC_FLAGS
        --default-stream per-thread
        -O3
)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -Wall -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -s")
# set(CUBLAS_STATIC_LIB /home/mazt/cuda-11.4.2/lib64/libcublas_static.a)
# set(CUBLAS_STATIC_PREFIX /home/mazt/cuda-11.4.2/lib64)
### tensor backend
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
if(CUDAToolkit_FOUND)

    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    # SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

    # 全局 CUDA 编译选项
    # include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDA_CUBLAS_DIRS} "src")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -Wall")
    # 全局 CUDA 链接选项
    set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -Wl,--gc-sections")
    
    list(APPEND LINKER_LIBS CUDA::cudart)#${CUDA_CUDART_LIBRARY} ${CUDA_npp_LIBRARY},  ${CUDA_cublas_LIBRARY}
    
    # include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} "src")
    # set(MODULE_NAMES broadcast_ops basic_tensor_ops judge_ops nn_ops device_info)
    
    # ###  paras.h
    # CUDA_ADD_LIBRARY(global_streams MODULE src/paras.cu OPTIONS ${ARCH_FLAGS})
    # # pybind11_add_module(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu)
    # target_link_libraries(global_streams ${CUDA_CUDART_LIBRARY})#paras
    # # add_library(global_streams MODULE src/paras.cu)

    # pybind11_extension(global_streams)
    # pybind11_strip(global_streams)
    # set_target_properties(global_streams
    #     PROPERTIES
    #     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    #     CXX_VISIBILITY_PRESET "hidden"
    #     CUDA_VISIBILITY_PRESET "hidden"
    #     LINK_FLAGS "-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections"
    #     # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
    # )

    
    # install(TARGETS paras DESTINATION  ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # install(FILES paras.h DESTINATION  ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # foreach(MODULE_NAME ${MODULE_NAMES})
    #     message("Current item: ${item}")
    #     CUDA_ADD_LIBRARY(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu OPTIONS ${ARCH_FLAGS})
    #     # pybind11_add_module(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu)
    #     target_link_libraries(${MODULE_NAME} ${LINKER_LIBS})#paras
    #     pybind11_extension(${MODULE_NAME})
    #     pybind11_strip(${MODULE_NAME})
    #     set_target_properties(${MODULE_NAME}
    #         PROPERTIES
    #         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    #         CXX_VISIBILITY_PRESET "hidden"
    #         CUDA_VISIBILITY_PRESET "hidden"
    #         LINK_FLAGS "-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections"
    #         # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
    #     )
        
    #     # install(TARGETS ${MODULE_NAME}
    #     # LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
    #     # )
    # endforeach()

    
    add_library(ndarray_cuda_backend MODULE ${CMAKE_CURRENT_SOURCE_DIR}/src/ndarray_cuda_backend.cu)
    target_link_libraries(ndarray_cuda_backend ${LINKER_LIBS} ${CUDA_cublas_LIBRARY})
        # ${CUBLAS_STATIC_LIB}
        # ${CUBLAS_STATIC_PREFIX}/libculibos.a
        # ${CUBLAS_STATIC_PREFIX}/libcublasLt_static.a
        # ${CUBLAS_STATIC_PREFIX}/libcudart_static.a)#paras ${CUDA_cublas_LIBRARY}
    set_target_properties(ndarray_cuda_backend
        PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        LINK_FLAGS "-Wl,-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections,--as-needed"
        # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
        LINKER_LANGUAGE CUDA
    )
    # target_link_options(ndarray_cuda_backend PRIVATE -gc-sections,-exclude-libs=ALL,-ffunction-sections,-fdata-sections)
    # 针对特定目标的编译选项
    target_compile_options(ndarray_cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
    # target_compile_options(ndarray_cuda_backend PRIVATE --compiler-options -Wextra)
    # target_link_options(ndarray_cuda_backend PRIVATE -Wl,--gc-sections,-exclude-libs=ALL)
    # pybind11_add_module(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu)
    pybind11_extension(ndarray_cuda_backend)
    pybind11_strip(ndarray_cuda_backend)
    
    add_library(pygp_utils MODULE src/utils_backend.cu)
    target_link_libraries(pygp_utils ${LINKER_LIBS} ${CUDA_cublas_LIBRARY})
        # ${CUBLAS_STATIC_LIB}
        # ${CUBLAS_STATIC_PREFIX}/libculibos.a
        # ${CUBLAS_STATIC_PREFIX}/libcublasLt_static.a
        # ${CUBLAS_STATIC_PREFIX}/libcudart_static.a)#paras ${CUDA_cublas_LIBRARY}
        
    set_target_properties(pygp_utils
        PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        LINK_FLAGS "-Wl,-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections,--as-needed,-O3"
        # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
        LINKER_LANGUAGE CUDA
    )
    pybind11_extension(pygp_utils)
    pybind11_strip(pygp_utils)

    

endif()

### executor backend

if(CUDAToolkit_FOUND)
    message(STATUS "Found cuda, building cuda backend")
    message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
    # include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDA_CUBLAS_DIRS} "src")
    list(APPEND LINKER_LIBS CUDA::cudart)#${CUDA_CUDART_LIBRARY}

    execute_process(COMMAND "nvidia-smi" ERROR_QUIET RESULT_VARIABLE NV_RET)
    # CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)

    add_library(executor MODULE ${CMAKE_CURRENT_SOURCE_DIR}/src/exec_cuda_backend.cu)

    target_link_libraries(executor ${LINKER_LIBS} CUDA::cublas)
        # ${CUBLAS_STATIC_LIB}
        # ${CUBLAS_STATIC_PREFIX}/libculibos.a
        # ${CUBLAS_STATIC_PREFIX}/libcublasLt_static.a
        # ${CUBLAS_STATIC_PREFIX}/libcudart_static.a)# CUDA::cublas 
    pybind11_extension(executor)
    pybind11_strip(executor)

    set_target_properties(executor
        PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        LINK_FLAGS "-Wl,-exclude-libs=ALL,--gc-sections,-ffunction-sections,-fdata-sections,--as-needed"
        LINKER_LANGUAGE CUDA
    )
    # target_link_options(executor PRIVATE -gc-sections,-exclude-libs=ALL,-ffunction-sections,-fdata-sections)
    # target_compile_options(executor PRIVATE --compiler-options -Wextra)
    target_compile_options(executor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
    # target_link_options(executor PRIVATE -Wl,--gc-sections,-exclude-libs=ALL)
else()
    message(STATUS "Can not found CUDA")
endif()
message(STATUS "Python3_INCLUDE_DIRS: ${CMAKE_INSTALL_LIBDIR}, ${CMAKE_CURRENT_SOURCE_DIR}")

# Install pygp_utils modules
install(TARGETS pygp_utils
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
        )

# Install pygp_cash modules
install(TARGETS pygp_cash
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
        )

# Install CUDA modules（if exists）
if(CUDAToolkit_FOUND)
    install(TARGETS executor
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
            )
endif()

add_custom_command(TARGET ndarray_cuda_backend POST_BUILD
    COMMAND strip $<TARGET_FILE:ndarray_cuda_backend>
)
add_custom_command(TARGET executor POST_BUILD
    COMMAND strip $<TARGET_FILE:executor>
)