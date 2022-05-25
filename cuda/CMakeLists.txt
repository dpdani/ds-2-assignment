cmake_minimum_required(VERSION 3.22)
project(ds-cuda CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)


add_executable(ds-cuda main.cu)
add_executable(cuda-test test.cu)

set_target_properties(ds-cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda-test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(ds-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -lcudadevrt
        --relocatable-device-code=true
        >)
target_compile_options(cuda-test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -lcudadevrt
        --relocatable-device-code=true
        >)

# Tests

enable_testing()

add_test(NAME nop COMMAND cuda-test nop)
add_test(NAME Host COMMAND cuda-test host)
add_test(NAME Device COMMAND cuda-test device)
add_test(NAME Device-Average-Path-Length COMMAND cuda-test device-path-length)
