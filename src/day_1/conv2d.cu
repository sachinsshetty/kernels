#include <stdio.h>
#include <assert.h>
__global__ void conv2d(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inH - kH + 1 && j < inW - kW + 1) {
        float sum = 0.0f;
        for (int ki = 0; ki < kH; ki++) {
            for (int kj = 0; kj < kW; kj++) {
                sum += input[(i + ki) * inW + (j + kj)] * kernel[ki * kW + kj];
            }
        }
        output[i * (inW - kW + 1) + j] = sum;
    }
}
int main() {
    int inW = 32, inH = 32, kW = 3, kH = 3;
    int outW = inW - kW + 1, outH = inH - kH + 1;
    int inSize = inW * inH * sizeof(float);
    int kSize = kW * kH * sizeof(float);
    int outSize = outW * outH * sizeof(float);
    // Host memory
    float *h_input = (float*)malloc(inSize);
    float *h_kernel = (float*)malloc(kSize);
    float *h_output = (float*)malloc(outSize);
    for (int i = 0; i < inW * inH; i++) h_input[i] = i % 10;
    for (int i = 0; i < kW * kH; i++) h_kernel[i] = 1.0f / 9.0f; // Average filter
    // Device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, inSize);
    cudaMalloc(&d_kernel, kSize);
    cudaMalloc(&d_output, outSize);
    cudaMemcpy(d_input, h_input, inSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kSize, cudaMemcpyHostToDevice);
    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outH + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2d<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output, inW, inH, kW, kH);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, outSize, cudaMemcpyDeviceToHost);
    // Verify (compare with CPU version)
    printf("Convolution complete!\n");
    // Cleanup
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    free(h_input); free(h_kernel); free(h_output);
    return 0;
}