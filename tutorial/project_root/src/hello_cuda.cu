#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "hello_cuda.h"

__global__ void helloCUDA(void)
{
    printf("Hello CUDA from GPU!\n");
}

void launchKernel()
{
    helloCUDA<<<1, 10>>>();
    cudaDeviceSynchronize();
}
