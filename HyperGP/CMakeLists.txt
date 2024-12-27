cmake_minimum_required(VERSION 3.2)

project(HyperGP_Tensor_backend)

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
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -march=native ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
else()
    set(CMAKE_CXX_FLAGS "/std:c++11 -O3 -march=native ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})

############################################################################
### CPU BACKEND
############################################################################

### tree2graph_transformer

add_library(HyperGP_utils MODULE src/utils_backend.cc)
target_link_libraries(HyperGP_utils PUBLIC ${LINKER_LIBS})
pybind11_extension(HyperGP_utils)
pybind11_strip(HyperGP_utils)

set_target_properties(HyperGP_utils
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    CXX_VISIBILITY_PRESET "hidden"
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET tree2graph_transformer PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

### cash_backend
add_library(HyperGP_cash MODULE src/cash_backend.cc)
target_link_libraries(HyperGP_cash PUBLIC ${LINKER_LIBS})
pybind11_extension(HyperGP_cash)
pybind11_strip(HyperGP_cash)

set_target_properties(HyperGP_cash
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    CXX_VISIBILITY_PRESET "hidden"    
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_property(TARGET HyperGP_cash PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()

############################################################################
### CUDA BACKEND
############################################################################

set(CUDA_NVCC_FLAGS
        --default-stream per-thread
        -O3
)

### tensor backend

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Found cuda, building cuda backend")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} "src")
    list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

    execute_process(COMMAND "nvidia-smi" ERROR_QUIET RESULT_VARIABLE NV_RET)
    # CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)

    CUDA_ADD_LIBRARY(ndarray_cuda_backend MODULE src/ndarray_cuda_backend.cu OPTIONS ${ARCH_FLAGS})

    target_link_libraries(ndarray_cuda_backend ${LINKER_LIBS} ${CUDA_cublas_LIBRARY} ${CUDA_npp_LIBRARY})
    pybind11_extension(ndarray_cuda_backend)
    pybind11_strip(ndarray_cuda_backend)

    set_target_properties(ndarray_cuda_backend
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
    )
endif()


### executor backend

find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "Found cuda, building cuda backend")
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} "src")
    list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

    execute_process(COMMAND "nvidia-smi" ERROR_QUIET RESULT_VARIABLE NV_RET)
    # CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)

    CUDA_ADD_LIBRARY(executor MODULE src/exec_cuda_backend.cu OPTIONS ${ARCH_FLAGS})

    target_link_libraries(executor ${LINKER_LIBS})
    pybind11_extension(executor)
    pybind11_strip(executor)

    set_target_properties(executor
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden"
    )
endif()