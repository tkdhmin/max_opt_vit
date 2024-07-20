project(HelloAgainCuda LANGUAGES CXX CUDA)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add the CUDA source files
set(CUDA_SOURCE_FILES src/hello_cuda.cu src/main.cu)

# Print the CUDA source files (for debugging purposes)
message(STATUS "CUDA source files: ${CUDA_SOURCE_FILES}")

# Specify the executable to be created from the CUDA source files
add_executable(my_exec ${CUDA_SOURCE_FILES})

# Add include directories
target_include_directories(my_exec PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link CUDA runtime
target_link_libraries(my_exec PRIVATE ${CUDA_CUDART_LIBRARY})

# Add Google Test
# Adjust the path to your actual Google Test location
add_subdirectory(${CMAKE_SOURCE_DIR}/external/googletest ${CMAKE_BINARY_DIR}/googletest)

# Add the test executable and link with CUDA and Google Test
add_executable(test_launch_kernel tests/test_launch_kernel.cpp src/hello_cuda.cu)
target_include_directories(test_launch_kernel PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_launch_kernel PRIVATE gtest gtest_main ${CUDA_CUDART_LIBRARY})

# Enable testing
enable_testing()
add_test(NAME test_launch_kernel COMMAND test_launch_kernel)
