#include <stdio.h>
__global__ void helloKernel() {
    printf("Hello from GPU thread %d, block %d\n", threadIdx.x, blockIdx.x);
}
int main() {
    helloKernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    return 0;
}