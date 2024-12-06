cmake_minimum_required(VERSION 3.12)
project(OpsModule LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

include_directories(/usr/local/cuda/include)

find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_80)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/src)

function(add_module MODULE_NAME)
    set(SOURCE_FILES
        src/${MODULE_NAME}/${MODULE_NAME}.cu
        src/${MODULE_NAME}/${MODULE_NAME}_wrapper.cpp
    )
    add_library(${MODULE_NAME} SHARED ${SOURCE_FILES})
    
    set_target_properties(${MODULE_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        PREFIX ""  
        SUFFIX ".so"  
    )
    
    target_link_libraries(${MODULE_NAME} PRIVATE pybind11::module)

    install(TARGETS ${MODULE_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endfunction()

add_module(conv2d)
add_module(matadd)
add_module(matmul)
add_module(pooling)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
