cmake_minimum_required(VERSION 3.16)
project(sop_nx_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/libraries were not found")
endif()

add_executable(sop_nx_infer src/main.cpp)
target_include_directories(sop_nx_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sop_nx_infer PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    Threads::Threads
    ${CUDA_CUDART_LIBRARY})
target_compile_options(sop_nx_infer PRIVATE -Wall -Wextra -Wpedantic -O3)

# Experimental two-station probe reuses the exact model/pre/postprocessing code.
add_executable(sop_nx_dual_probe src/dual_probe.cpp)
target_include_directories(sop_nx_dual_probe PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sop_nx_dual_probe PRIVATE ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} Threads::Threads ${CUDA_CUDART_LIBRARY})
target_compile_options(sop_nx_dual_probe PRIVATE -Wall -Wextra -Wpedantic -O3)

include_directories(${TENSORRT_INCLUDE_DIR})
cuda_add_library(sop_silu_plugin SHARED src/silu_plugin.cu OPTIONS -std=c++14
    -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_72,code=compute_72)
target_link_libraries(sop_silu_plugin ${TENSORRT_LIBRARY} ${CUDA_CUDART_LIBRARY})
add_executable(sop_nx_joint_probe src/joint_probe.cpp)
target_include_directories(sop_nx_joint_probe PRIVATE
    ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sop_nx_joint_probe PRIVATE sop_silu_plugin ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY} Threads::Threads ${CUDA_CUDART_LIBRARY})
target_compile_options(sop_nx_joint_probe PRIVATE -Wall -Wextra -Wpedantic -O3)

include(CTest)
if(BUILD_TESTING)
    add_executable(silu_plugin_test tests/silu_plugin_test.cpp)
    target_include_directories(silu_plugin_test PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(silu_plugin_test PRIVATE sop_silu_plugin ${TENSORRT_LIBRARY} ${CUDA_CUDART_LIBRARY})
    add_test(NAME silu_plugin COMMAND silu_plugin_test)
    set_tests_properties(silu_plugin PROPERTIES TIMEOUT 20)
    add_executable(pipeline_queue_test tests/pipeline_queue_test.cpp)
    target_include_directories(pipeline_queue_test PRIVATE src)
    target_link_libraries(pipeline_queue_test PRIVATE Threads::Threads)
    add_test(NAME pipeline_queue COMMAND pipeline_queue_test)
    set_tests_properties(pipeline_queue PROPERTIES TIMEOUT 15)
endif()


option(SOP_BUILD_ACCEL "Build experimental hardware decode and concurrent inference" OFF)
if(SOP_BUILD_ACCEL)
    include(accel.cmake)
endif()
