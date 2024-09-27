# cmake_minimum_required(VERSION 3.16)
project(VectorSum LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CUDA_SOURCE_FILE vector_sum.cu)

# Specify the executable to be created from the CUDA source file
add_executable(vector_sum ${CUDA_SOURCE_FILE})
