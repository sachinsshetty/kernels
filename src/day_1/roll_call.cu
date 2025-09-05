#include <stdio.h>

__global__ void roll_call() {
    int threadIndex = threadIdx.x;
    printf("Thread %d here!\n", threadIndex);
}

int main() {
    roll_call<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}

// nvcc roll_call.cu -o roll_call