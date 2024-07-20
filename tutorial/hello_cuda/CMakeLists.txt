# cmake_minimum_required(VERSION 3.16)
project(HelloCuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CUDA_SOURCE_FILE main.cu)

# Specify the executable to be created from the CUDA source file
add_executable(main ${CUDA_SOURCE_FILE})