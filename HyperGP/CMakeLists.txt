cmake_minimum_required(VERSION 3.2)

project(PyGP_Tensor_backend)

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

### tree2graph_transformer

add_library(pygp_utils MODULE src/utils_backend.cc)
target_link_libraries(pygp_utils ${LINKER_LIBS})
pybind11_extension(pygp_utils)
pybind11_strip(pygp_utils)

set_target_properties(pygp_utils
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    CXX_VISIBILITY_PRESET "hidden"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET tree2graph_transformer PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

### cash_backend
add_library(pygp_cash MODULE src/cash_backend.cc)
target_link_libraries(pygp_cash ${LINKER_LIBS})
pybind11_extension(pygp_cash)
pybind11_strip(pygp_cash)

set_target_properties(pygp_cash
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    CXX_VISIBILITY_PRESET "hidden"    
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

### tensor backend
find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} "src")
    list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY})#${CUDA_npp_LIBRARY}
    set(MODULE_NAMES broadcast_ops basic_tensor_ops judge_ops nn_ops device_info)

    
    ###  paras.h
    CUDA_ADD_LIBRARY(global_streams MODULE src/paras.cu OPTIONS ${ARCH_FLAGS})
    # pybind11_add_module(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu)
    target_link_libraries(global_streams ${CUDA_CUDART_LIBRARY})#paras
    # add_library(global_streams MODULE src/paras.cu)

    pybind11_extension(global_streams)
    pybind11_strip(global_streams)
    set_target_properties(global_streams
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
        # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
    )
    # install(TARGETS paras DESTINATION  ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # install(FILES paras.h DESTINATION  ${CMAKE_CURRENT_SOURCE_DIR}/src)

    foreach(MODULE_NAME ${MODULE_NAMES})
        message("Current item: ${item}")
        CUDA_ADD_LIBRARY(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu OPTIONS ${ARCH_FLAGS})
        # pybind11_add_module(${MODULE_NAME} MODULE src/ndarray_cuda_backend.cu)
        target_link_libraries(${MODULE_NAME} ${LINKER_LIBS})#paras
        pybind11_extension(${MODULE_NAME})
        pybind11_strip(${MODULE_NAME})
        set_target_properties(${MODULE_NAME}
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
            CXX_VISIBILITY_PRESET "hidden"
            CUDA_VISIBILITY_PRESET "hidden"
            # SUFFIX ${CMAKE_SHARED_MODULE_SUFFIX}  # 跨平台支持
        )
        
        install(TARGETS ${MODULE_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
        )
    endforeach()
endif()

### executor backend

if(CUDA_FOUND)
    message(STATUS "Found cuda, building cuda backend")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} "src")
    list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

    execute_process(COMMAND "nvidia-smi" ERROR_QUIET RESULT_VARIABLE NV_RET)
    # CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)

    CUDA_ADD_LIBRARY(executor MODULE src/exec_cuda_backend.cu OPTIONS ${ARCH_FLAGS})

    target_link_libraries(executor ${LINKER_LIBS} ${CUDA_cublas_LIBRARY})
    pybind11_extension(executor)
    pybind11_strip(executor)

    set_target_properties(executor
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
    )
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
if(CUDA_FOUND)
    install(TARGETS executor
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/HyperGP/src
            )
endif()