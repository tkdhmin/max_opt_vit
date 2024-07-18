#include <stdio.h>
#include "hello_cuda.h"

int main(void)
{
    printf("Hello GPU from CPU!\n");
    launchKernel();
    return 0;
}
