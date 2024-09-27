#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

// The size of the vector
#define NUM_DATA 1024

// Simple vector sum gpu kernel
__global__ void VecAddOnDevice(int *_c, int *_a, int *_b) {
  // int tID = threadIdx.x;
  int tID = blockIdx.x * blockDim.x + threadIdx.x; // 글로벌 인덱스 계산
  if (tID < NUM_DATA)
    _c[tID] = _a[tID] + _b[tID];
}

/**
 * @brief Initialize host memory
 *
 * @param memory
 * @param size
 */
void HostMemInit(int *&memory, int size) {
  assert(size > 0);

  if (memory) {
    std::cout << "free" << std::endl;

    delete[] memory;
    memory = nullptr;
  }

  int memSize = sizeof(int) * size;
  // std::cout << NUM_DATA << " elements, " << memSize << " Bytes" << std::endl;

  memory = new int[size];
  memset(memory, 0, memSize);
}

/**
 * @brief Initialize device memory
 *
 * @param memory Pointer to the device memory to initialize
 * @param size Number of elements in the memory
 */
void DeviceMemInit(int *&memory, int size) {
  int memSize = sizeof(int) * size;
  cudaError_t err = cudaMalloc(&memory, memSize);
  if (err != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaMemset(memory, 0, memSize);
}

/**
 * @brief Return a random digit from 0 to 9
 *
 * @return int
 */
int RandDataGen() { return rand() % 10; }

/**
 * @brief Generate random data and fill the memory
 *
 * @param memory
 * @param size
 */
void DataPreFill(int *&memory, int size) {
  for (int i = 0; i < size; i++)
    memory[i] = RandDataGen();
}

/**
 * @brief Add two vectors and store the output to results on the host.
 *
 * @param result
 * @param vec_a
 * @param vec_b
 * @param size
 */
void VecAddOnHost(int *result, const int *vec_a, const int *vec_b, int size) {
  for (int i = 0; i < size; i++)
    result[i] = vec_a[i] + vec_b[i];
}

/**
 * @brief Check if two vectors are equal
 *
 * @param vec Pointer to the first vector
 * @param other Pointer to the second vector
 * @param size Number of elements in the vectors
 * @return true if the vectors are equal, false otherwise
 */
bool CheckVectorEqual(int *vec, int *other, int size) {
  for (int i = 0; i < size; i++) {
    if (vec[i] != other[i]) {
      std::cout << vec[i] << " != " << other[i] << " at index " << i
                << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief Entry function
 *
 * @return int
 */
int main() {
  int *leading_vector = nullptr, *last_vector = nullptr;
  int *a = nullptr, *b = nullptr, *c = nullptr, *h_c = nullptr;
  int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  // Data gen
  // auto startCPUdatagen = std::chrono::high_resolution_clock::now();
  HostMemInit(leading_vector, NUM_DATA);
  HostMemInit(last_vector, NUM_DATA);
  DataPreFill(leading_vector, NUM_DATA);
  DataPreFill(last_vector, NUM_DATA);
  // auto endCPUdatagen = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::milli> durationCPUdatagen =
  // endCPUdatagen - startCPUdatagen; std::cout << "The elapsed time for
  // generating data: " << durationCPUdatagen.count() << " ms" << std::endl;

  // Host-side computation
  // Memory allocation on the host-side
  HostMemInit(a, NUM_DATA);
  memcpy(a, leading_vector, NUM_DATA * sizeof(int));
  HostMemInit(b, NUM_DATA);
  memcpy(b, last_vector, NUM_DATA * sizeof(int));
  HostMemInit(h_c, NUM_DATA);

  VecAddOnHost(h_c, a, b, NUM_DATA);

  // Device-side computation
  // Memory allocation to the device-side
  HostMemInit(c, NUM_DATA);
  DeviceMemInit(d_a, NUM_DATA);
  DeviceMemInit(d_b, NUM_DATA);
  DeviceMemInit(d_c, NUM_DATA);

  // Data Copy
  cudaMemcpy(d_a, a, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(int) * NUM_DATA, cudaMemcpyHostToDevice);

  // Kernel call
  VecAddOnDevice<<<1, NUM_DATA>>>(d_c, d_a, d_b);

  // Data copy-back
  cudaMemcpy(c, d_c, sizeof(int) * NUM_DATA, cudaMemcpyDeviceToHost);

  // Release device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // Check the integrity of an output.
  bool res = CheckVectorEqual(c, h_c, NUM_DATA);

  std::cout << "Vector Sum Verification: ";
  if (res)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;

  // Release host memory
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] leading_vector;
  delete[] last_vector;
  delete[] h_c;

  return 0;
}
