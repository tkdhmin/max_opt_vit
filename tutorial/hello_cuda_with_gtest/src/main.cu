#include "hello_cuda.h"
#include <stdio.h>

int main(void) {
  printf("Hello GPU from CPU!\n");
  launchKernel();
  return 0;
}
