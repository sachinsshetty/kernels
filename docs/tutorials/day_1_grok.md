Learning CUDA with C++ and implementing AlexNet in a 3-day intensive course is an ambitious goal. AlexNet is a deep convolutional neural network (CNN) that requires understanding both CUDA programming and deep learning concepts. Below, I outline a structured 3-day plan to learn CUDA and implement a simplified version of AlexNet. This plan assumes you have basic C++ knowledge and some familiarity with neural networks. If you're new to deep learning, the focus will be on CUDA implementation, with AlexNet as a practical project.

---

### Prerequisites
- **Hardware**: NVIDIA GPU (CUDA-capable, e.g., GTX 1050 or better).
- **Software**: 
  - CUDA Toolkit (v11.x or later, install from NVIDIA's website).
  - C++ compiler (e.g., GCC, MSVC).
  - CMake (for building the project).
  - Optional: cuDNN for optimized convolution operations (if time permits).
- **Resources**:
  - NVIDIA CUDA C Programming Guide (online).
  - AlexNet paper (Krizhevsky et al., 2012).
  - Basic image processing library (e.g., OpenCV or stb_image for loading data).
- **Dataset**: Use a small subset of ImageNet or CIFAR-10 for simplicity (downloadable from Kaggle or torchvision).

---

### Day 1: CUDA Fundamentals and Setup
**Goal**: Learn CUDA basics, set up the environment, and implement simple matrix operations.

#### Morning (3-4 hours): CUDA Basics
1. **Understand CUDA Architecture**:
   - Read: CUDA Programming Guide (Sections 1-2).
   - Key concepts: GPU threads, blocks, grids, memory hierarchy (global, shared, registers).
   - Watch: NVIDIA’s “CUDA Basics” video (YouTube, ~30 mins).

2. **Set Up Environment**:
   - Install CUDA Toolkit and verify with `nvcc --version`.
   - Write a simple CUDA “Hello World” program:
     ```cpp
     #include <stdio.h>
     __global__ void helloKernel() {
         printf("Hello from GPU thread %d\n", threadIdx.x);
     }
     int main() {
         helloKernel<<<1, 10>>>();
         cudaDeviceSynchronize();
         return 0;
     }
     ```
   - Compile: `nvcc hello.cu -o hello` and run.

3. **Matrix Addition in CUDA**:
   - Implement a kernel for matrix addition (2D arrays).
   - Example:
     ```cpp
     __global__ void matrixAdd(float *A, float *B, float *C, int N) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         int j = blockIdx.y * blockDim.y + threadIdx.y;
         if (i < N && j < N) {
             C[i * N + j] = A[i * N + j] + B[i * N + j];
         }
     }
     ```
   - Practice memory allocation (`cudaMalloc`, `cudaMemcpy`) and kernel launches.

#### Afternoon (3-4 hours): Convolution Basics
1. **Understand Convolution**:
   - Study 2D convolution (core operation in AlexNet).
   - Read: CS231n notes on CNNs (Stanford, online).
   - Write a CPU-based 2D convolution in C++:
     ```cpp
     void conv2d(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH) {
         for (int i = 0; i < inH - kH + 1; i++) {
             for (int j = 0; j < inW - kW + 1; j++) {
                 float sum = 0.0f;
                 for (int ki = 0; ki < kH; ki++) {
                     for (int kj = 0; kj < kW; kj++) {
                         sum += input[(i + ki) * inW + (j + kj)] * kernel[ki * kW + kj];
                     }
                 }
                 output[i * (inW - kW + 1) + j] = sum;
             }
         }
     }
     ```

2. **Port Convolution to CUDA**:
   - Convert the above to a CUDA kernel.
   - Use shared memory to optimize memory access.
   - Example:
     ```cpp
     __global__ void convolute(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH) {
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
     ```
   - Test with a small image (e.g., 32x32) and a 3x3 kernel.

#### Evening (1-2 hours): Project Setup
- Set up a CMake project for AlexNet.
- Download CIFAR-10 dataset and write code to load images (use stb_image or OpenCV).
- Plan AlexNet layers: 5 conv layers, 3 fully connected layers, ReLU, max-pooling, dropout.

---

### Day 2: Implement AlexNet Layers in CUDA
**Goal**: Implement core AlexNet components (convolution, pooling, ReLU) in CUDA.

#### Morning (3-4 hours): Convolution and ReLU
1. **Optimize Convolution Kernel**:
   - Use shared memory to load input tiles.
   - Handle multiple input/output channels (AlexNet has 3 input channels for RGB).
   - Example (multi-channel convolution):
     ```cpp
     __global__ void conv2dMulti(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH, int inC, int outC) {
         int i = blockIdx.y * blockDim.y + threadIdx.y;
         int j = blockIdx.x * blockDim.x + threadIdx.x;
         int oc = blockIdx.z; // Output channel
         if (i < inH - kH + 1 && j < inW - kW + 1 && oc < outC) {
             float sum = 0.0f;
             for (int c = 0; c < inC; c++) {
                 for (int ki = 0; ki < kH; ki++) {
                     for (int kj = 0; kj < kW; kj++) {
                         sum += input[c * inW * inH + (i + ki) * inW + (j + kj)] *
                                kernel[oc * inC * kW * kH + c * kW * kH + ki * kW + kj];
                     }
                 }
             }
             output[oc * (inW - kW + 1) * (inH - kH + 1) + i * (inW - kW + 1) + j] = sum;
         }
     }
     ```

2. **ReLU Activation**:
   - Implement a simple CUDA kernel for ReLU (`max(0, x)`).
   - Example:
     ```cpp
     __global__ void relu(float *input, float *output, int size) {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         if (idx < size) {
             output[idx] = input[idx] > 0 ? input[idx] : 0;
         }
     }
     ```

#### Afternoon (3-4 hours): Pooling and Fully Connected Layers
1. **Max-Pooling**:
   - Implement a 2x2 max-pooling kernel with stride 2.
   - Example:
     ```cpp
     __global__ void maxPool(float *input, float *output, int inW, int inH, int poolSize, int stride) {
         int i = blockIdx.y * blockDim.y + threadIdx.y;
         int j = blockIdx.x * blockDim.x + threadIdx.x;
         int c = blockIdx.z; // Channel
         if (i < inH / stride && j < inW / stride) {
             float maxVal = input[c * inW * inH + i * stride * inW + j * stride];
             for (int pi = 0; pi < poolSize; pi++) {
                 for (int pj = 0; pj < poolSize; pj++) {
                     float val = input[c * inW * inH + (i * stride + pi) * inW + (j * stride + pj)];
                     maxVal = max(maxVal, val);
                 }
             }
             output[c * (inW / stride) * (inH / stride) + i * (inW / stride) + j] = maxVal;
         }
     }
     ```

2. **Fully Connected Layer**:
   - Treat as matrix multiplication (use cuBLAS if time permits, or write a simple kernel).
   - Example (naive matrix multiplication):
     ```cpp
     __global__ void matMul(float *A, float *B, float *C, int M, int N, int K) {
         int i = blockIdx.y * blockDim.y + threadIdx.y;
         int j = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < M && j < N) {
             float sum = 0.0f;
             for (int k = 0; k < K; k++) {
                 sum += A[i * K + k] * B[k * N + j];
             }
             C[i * N + j] = sum;
         }
     }
     ```

#### Evening (1-2 hours): AlexNet Architecture
- Define AlexNet structure (simplified):
  - Conv1: 96 filters, 11x11, stride 4, ReLU, max-pool (3x3, stride 2).
  - Conv2: 256 filters, 5x5, ReLU, max-pool.
  - Conv3: 384 filters, 3x3, ReLU.
  - Conv4: 384 filters, 3x3, ReLU.
  - Conv5: 256 filters, 3x3, ReLU, max-pool.
  - FC1: 4096 units, ReLU, dropout.
  - FC2: 4096 units, ReLU, dropout.
  - FC3: 1000 units (or 10 for CIFAR-10), softmax.
- Write a forward-pass function combining these layers.

---

### Day 3: Training and Optimization
**Goal**: Implement training (backpropagation) and optimize performance.

#### Morning (3-4 hours): Backpropagation
1. **Understand Backpropagation**:
   - Read: CS231n notes on backpropagation.
   - Focus on gradients for convolution, ReLU, pooling, and fully connected layers.

2. **Implement Gradient Kernels**:
   - Convolution gradients (for weights and inputs).
   - Example (weight gradient for convolution):
     ```cpp
     __global__ void convWeightGrad(float *input, float *gradOutput, float *gradKernel, int inW, int inH, int kW, int kH, int inC, int outC) {
         int oc = blockIdx.z; // Output channel
         int c = blockIdx.y; // Input channel
         int ki = threadIdx.y;
         int kj = threadIdx.x;
         if (oc < outC && c < inC && ki < kH && kj < kW) {
             float sum = 0.0f;
             for (int i = 0; i < inH - kH + 1; i++) {
                 for (int j = 0; j < inW - kW + 1; j++) {
                     sum += input[c * inW * inH + (i + ki) * inW + (j + kj)] *
                            gradOutput[oc * (inW - kW + 1) * (inH - kH + 1) + i * (inW - kW + 1) + j];
                 }
             }
             gradKernel[oc * inC * kW * kH + c * kW * kH + ki * kW + kj] = sum;
         }
     }
     ```

3. **ReLU and Pooling Gradients**:
   - ReLU: Pass gradient if input > 0.
   - Max-pooling: Pass gradient to max location.

#### Afternoon (3-4 hours): Training Loop
1. **Implement Training**:
   - Write a training loop: forward pass, compute loss (e.g., cross-entropy), backward pass, update weights (SGD).
   - Example (SGD update):
     ```cpp
     __global__ void sgdUpdate(float *weights, float *grads, float lr, int size) {
         int idx = blockIdx.x * blockDim.x + threadIdx.x;
         if (idx < size) {
             weights[idx] -= lr * grads[idx];
         }
     }
     ```

2. **Test on CIFAR-10**:
   - Train on a small batch (e.g., 100 images).
   - Monitor loss and accuracy.

#### Evening (1-2 hours): Optimization and Debugging
- **Optimize CUDA Kernels**:
  - Use shared memory for convolution.
  - Experiment with block sizes (e.g., 16x16 or 32x32).
  - Profile with NVIDIA Nsight or `nvprof`.
- **Debug**:
  - Check intermediate outputs (e.g., conv layer outputs).
  - Compare with a CPU implementation for correctness.

---

### Notes
- **Simplifications**: Due to time constraints, skip dropout and use a smaller dataset (CIFAR-10). Ignore cuDNN unless you’re comfortable with it.
- **Challenges**: CUDA memory management and debugging are tricky. Use `cudaDeviceSynchronize` and check errors with `cudaGetLastError`.
- **Resources**:
  - CUDA samples (included with CUDA Toolkit).
  - Online tutorials (e.g., Mark Harris’s CUDA blog series).
  - AlexNet PyTorch implementation (for reference, not copying).
- **Post-Course**: Study cuDNN, optimize kernels further, and scale to full ImageNet.

This plan is intense but feasible for learning CUDA basics and implementing a simplified AlexNet. Focus on understanding CUDA memory and parallelism while building the network. If you hit roadblocks, simplify (e.g., fewer layers) or ask for specific help!

---


Learning CUDA with C++ in a 3-day intensive course while implementing AlexNet is a challenging but achievable goal. Below is a detailed, step-by-step plan to learn CUDA programming from scratch and work toward implementing a simplified version of AlexNet. This guide assumes you have basic C++ knowledge and some understanding of neural networks. If you're new to deep learning, the focus will be on CUDA programming, using AlexNet as a practical application. Each day includes theory, hands-on coding, and resources, with clear steps to follow.

---

### Prerequisites
Before starting, ensure you have the following:
- **Hardware**: A CUDA-capable NVIDIA GPU (e.g., GTX 1050, RTX series). Check compatibility on NVIDIA’s CUDA GPUs page.
- **Software**:
  - **CUDA Toolkit** (v11.x or later, download from NVIDIA’s website).
  - **C++ Compiler**: GCC (Linux), MSVC (Windows), or Clang.
  - **CMake**: For project organization.
  - **Optional**: cuDNN (for optimized convolution, but we’ll keep it simple initially).
  - **Image Processing**: stb_image (lightweight, header-only) or OpenCV for loading images.
- **Dataset**: CIFAR-10 (simpler than ImageNet, ~60,000 32x32 images, downloadable from Kaggle or torchvision).
- **Resources**:
  - NVIDIA CUDA C Programming Guide (online, free).
  - AlexNet paper (Krizhevsky et al., 2012, available via Google Scholar).
  - CS231n CNN notes (Stanford, online).
  - CUDA by Example (Sanders & Kandrot, book or PDF).
- **Setup**:
  - Install CUDA Toolkit and verify: `nvcc --version`.
  - Install CMake and a C++ compiler.
  - Download CIFAR-10 and test image loading with stb_image or OpenCV.

---

### Day 1: CUDA Fundamentals and Basic Operations
**Goal**: Understand CUDA basics, set up the environment, and implement matrix operations as a foundation for neural networks.

#### Morning (9:00 AM - 12:30 PM): CUDA Basics
1. **Step 1: Learn CUDA Architecture (1 hour)**  
   - **Objective**: Understand how GPUs work and CUDA’s programming model.  
   - **Tasks**:
     - Read CUDA Programming Guide (Sections 1-2, ~20 pages) or watch NVIDIA’s “CUDA Basics” video (~30 mins, YouTube).
     - Key concepts to grasp:
       - **Threads, Blocks, Grids**: CUDA organizes threads in blocks (e.g., 32x32 threads) and grids (collections of blocks).
       - **Memory Hierarchy**: Global memory (slow, large), shared memory (fast, per-block), registers (fastest, per-thread).
       - **Kernel**: A function executed on the GPU by many threads in parallel.
     - Example: A grid of 2 blocks, each with 256 threads, can process 512 elements simultaneously.
   - **Action**: Take notes on threadIdx, blockIdx, blockDim, and gridDim.

2. **Step 2: Set Up CUDA Environment (30 mins)**  
   - **Objective**: Write and run your first CUDA program.  
   - **Tasks**:
     - Create a file `hello.cu`:
       ```cpp
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
       ```
     - Compile: `nvcc hello.cu -o hello`.
     - Run: `./hello` (Linux) or `hello.exe` (Windows).
     - Expected output: 8 lines (2 blocks x 4 threads) with thread and block IDs.
   - **Action**: Verify the program runs without errors. If it fails, check CUDA installation and GPU compatibility.

3. **Step 3: Matrix Addition in CUDA (2 hours)**  
   - **Objective**: Implement a simple CUDA kernel to add two matrices, introducing memory management.  
   - **Tasks**:
     - Create `matrix_add.cu`:
       ```cpp
       #include <stdio.h>
       #include <assert.h>
       __global__ void matrixAdd(float *A, float *B, float *C, int N) {
           int i = blockIdx.y * blockDim.y + threadIdx.y;
           int j = blockIdx.x * blockDim.x + threadIdx.x;
           if (i < N && j < N) {
               C[i * N + j] = A[i * N + j] + B[i * N + j];
           }
       }
       int main() {
           int N = 4; // 4x4 matrices
           int size = N * N * sizeof(float);
           float *h_A = (float*)malloc(size);
           float *h_B = (float*)malloc(size);
           float *h_C = (float*)malloc(size);
           // Initialize matrices
           for (int i = 0; i < N * N; i++) {
               h_A[i] = i; h_B[i] = i * 2; h_C[i] = 0;
           }
           // Device memory
           float *d_A, *d_B, *d_C;
           cudaMalloc(&d_A, size);
           cudaMalloc(&d_B, size);
           cudaMalloc(&d_C, size);
           cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
           cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
           // Launch kernel
           dim3 threadsPerBlock(2, 2);
           dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
           matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
           cudaDeviceSynchronize();
           cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
           // Verify
           for (int i = 0; i < N * N; i++) {
               assert(h_C[i] == h_A[i] + h_B[i]);
           }
           printf("Matrix addition successful!\n");
           // Cleanup
           cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
           free(h_A); free(h_B); free(h_C);
           return 0;
       }
       ```
     - Compile and run: `nvcc matrix_add.cu -o matrix_add`.
     - Understand each line: `cudaMalloc` allocates GPU memory, `cudaMemcpy` transfers data, `dim3` defines grid/block sizes.
   - **Action**: Test with different matrix sizes (e.g., 16x16). Debug any errors using `cudaGetLastError`.

#### Afternoon (1:30 PM - 5:30 PM): Convolution Basics
4. **Step 4: Understand Convolution (1 hour)**  
   - **Objective**: Learn 2D convolution, a core operation in AlexNet.  
   - **Tasks**:
     - Read CS231n notes on CNNs (section on convolution, ~30 mins).
     - Key concepts:
       - Convolution slides a kernel (e.g., 3x3) over an input image to produce a feature map.
       - Parameters: Kernel size, stride, padding.
       - Example: A 32x32 image with a 3x3 kernel (stride 1, no padding) produces a 30x30 output.
     - Write a CPU-based 2D convolution:
       ```cpp
       void conv2d(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH) {
           for (int i = 0; i < inH - kH + 1; i++) {
               for (int j = 0; j < inW - kW + 1; j++) {
                   float sum = 0.0f;
                   for (int ki = 0; ki < kH; ki++) {
                       for (int kj = 0; kj < kW; kj++) {
                           sum += input[(i + ki) * inW + (j + kj)] * kernel[ki * kW + kj];
                       }
                   }
                   output[i * (inW - kW + 1) + j] = sum;
               }
           }
       }
       ```
     - Test with a 5x5 image and 3x3 kernel (initialize randomly).
   - **Action**: Verify output manually for a small case (e.g., 3x3 image, 2x2 kernel).

5. **Step 5: Port Convolution to CUDA (2 hours)**  
   - **Objective**: Write a CUDA kernel for 2D convolution.  
   - **Tasks**:
     - Create `conv2d.cu`:
       ```cpp
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
       ```
     - Compile and run: `nvcc conv2d.cu -o conv2d`.
     - Compare output with CPU version for correctness.
   - **Action**: Experiment with different kernel sizes (e.g., 5x5) and input sizes.

6. **Step 6: Introduction to Shared Memory (1 hour)**  
   - **Objective**: Optimize convolution using shared memory.  
   - **Tasks**:
     - Read CUDA Programming Guide (Section 3.2.3 on shared memory).
     - Modify the convolution kernel to use shared memory for input tiles:
       ```cpp
       __global__ void conv2dShared(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH) {
           extern __shared__ float s_input[];
           int i = blockIdx.y * blockDim.y + threadIdx.y;
           int j = blockIdx.x * blockDim.x + threadIdx.x;
           int ti = threadIdx.y, tj = threadIdx.x;
           // Load input tile into shared memory
           int tileW = blockDim.x + kW - 1;
           if (i < inH && j < inW) {
               s_input[ti * tileW + tj] = input[i * inW + j];
           } else {
               s_input[ti * tileW + tj] = 0.0f;
           }
           __syncthreads();
           if (i < inH - kH + 1 && j < inW - kW + 1 && ti < blockDim.y && tj < blockDim.x) {
               float sum = 0.0f;
               for (int ki = 0; ki < kH; ki++) {
                   for (int kj = 0; kj < kW; kj++) {
                       sum += s_input[(ti + ki) * tileW + (tj + kj)] * kernel[ki * kW + kj];
                   }
               }
               output[i * (inW - kW + 1) + j] = sum;
           }
       }
       ```
     - Update main to allocate shared memory: `conv2dShared<<<numBlocks, threadsPerBlock, tileW * tileH * sizeof(float)>>>(...)`.
     - Test and compare performance (use `nvprof` or `nvidia-smi`).
   - **Action**: Measure execution time with `cudaEvent` API for baseline vs. shared memory.

#### Evening (6:30 PM - 8:30 PM): Project Setup
7. **Step 7: Set Up AlexNet Project (2 hours)**  
   - **Objective**: Create a project structure and load CIFAR-10 data.  
   - **Tasks**:
     - Create a CMake project:
       ```cmake
       cmake_minimum_required(VERSION 3.10)
       project(AlexNetCUDA LANGUAGES CXX CUDA)
       set(CMAKE_CUDA_STANDARD 11)
       add_executable(alexnet main.cu conv.cu relu.cu pool.cu fc.cu)
       set_target_properties(alexnet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
       ```
     - Download CIFAR-10 and write a data loader using stb_image:
       ```cpp
       #include "stb_image.h"
       float* loadImage(const char* path, int* w, int* h, int* c) {
           int channels;
           unsigned char* img = stbi_load(path, w, h, &channels, 3);
           float* data = (float*)malloc((*w) * (*h) * 3 * sizeof(float));
           for (int i = 0; i < (*w) * (*h) * 3; i++) {
               data[i] = img[i] / 255.0f; // Normalize to [0,1]
           }
           stbi_image_free(img);
           return data;
       }
       ```
     - Plan AlexNet architecture (simplified for CIFAR-10):
       - Input: 32x32x3 (RGB).
       - Conv1: 64 filters, 5x5, stride 1, ReLU, max-pool (2x2, stride 2).
       - Conv2: 64 filters, 5x5, ReLU, max-pool.
       - FC1: 384 units, ReLU.
       - FC2: 192 units, ReLU.
       - FC3: 10 units (for 10 classes), softmax.
   - **Action**: Load one CIFAR-10 image and verify dimensions.

---

### Day 2: Implement AlexNet Layers in CUDA
**Goal**: Implement core AlexNet components (convolution, ReLU, pooling, fully connected layers).

#### Morning (9:00 AM - 12:30 PM): Convolution and ReLU
8. **Step 8: Multi-Channel Convolution (2 hours)**  
   - **Objective**: Extend convolution to handle multiple input/output channels (e.g., RGB input, multiple filters).  
   - **Tasks**:
     - Write a multi-channel convolution kernel:
       ```cpp
       __global__ void conv2dMulti(float *input, float *kernel, float *output, int inW, int inH, int kW, int kH, int inC, int outC) {
           int i = blockIdx.y * blockDim.y + threadIdx.y;
           int j = blockIdx.x * blockDim.x + threadIdx.x;
           int oc = blockIdx.z; // Output channel
           if (i < inH - kH + 1 && j < inW - kW + 1 && oc < outC) {
               float sum = 0.0f;
               for (int c = 0; c < inC; c++) {
                   for (int ki = 0; ki < kH; ki++) {
                       for (int kj = 0; kj < kW; kj++) {
                           sum += input[c * inW * inH + (i + ki) * inW + (j + kj)] *
                                  kernel[oc * inC * kW * kH + c * kW * kH + ki * kW + kj];
                       }
                   }
               }
               output[oc * (inW - kW + 1) * (inH - kH + 1) + i * (inW - kW + 1) + j] = sum;
           }
       }
       ```
     - Update main to test with 3 input channels (RGB) and 64 output channels (Conv1).
     - Initialize kernels randomly (e.g., using `rand()`).
   - **Action**: Test with a CIFAR-10 image (32x32x3) and verify output size (28x28x64 for 5x5 kernel).

9. **Step 9: ReLU Activation (1 hour)**  
   - **Objective**: Implement ReLU activation (`max(0, x)`) in CUDA.  
   - **Tasks**:
     - Write a ReLU kernel:
       ```cpp
       __global__ void relu(float *input, float *output, int size) {
           int idx = blockIdx.x * blockDim.x + threadIdx.x;
           if (idx < size) {
               output[idx] = input[idx] > 0 ? input[idx] : 0;
           }
       }
       ```
     - Chain with convolution: Pass convolution output to ReLU.
     - Test with a small array and verify all negative values become 0.
   - **Action**: Integrate ReLU into the forward pass after Conv1.

#### Afternoon (1:30 PM - 5:30 PM): Pooling and Fully Connected Layers
10. **Step 10: Max-Pooling (2 hours)**  
    - **Objective**: Implement 2x2 max-pooling with stride 2.  
    - **Tasks**:
      - Write a max-pooling kernel:
        ```cpp
        __global__ void maxPool(float *input, float *output, int inW, int inH, int poolSize, int stride) {
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            int c = blockIdx.z; // Channel
            if (i < inH / stride && j < inW / stride) {
                float maxVal = input[c * inW * inH + i * stride * inW + j * stride];
                for (int pi = 0; pi < poolSize; pi++) {
                    for (int pj = 0; pj < poolSize; pj++) {
                        float val = input[c * inW * inH + (i * stride + pi) * inW + (j * stride + pj)];
                        maxVal = max(maxVal, val);
                    }
                }
                output[c * (inW / stride) * (inH / stride) + i * (inW / stride) + j] = maxVal;
            }
        }
        ```
      - Test with Conv1 output (28x28x64) and verify output size (14x14x64 for 2x2 pool, stride 2).
      - Use 3D grid for channels (blockIdx.z).
    - **Action**: Chain Conv1 → ReLU → MaxPool and verify dimensions.

11. **Step 11: Fully Connected Layer (2 hours)**  
    - **Objective**: Implement a fully connected layer as matrix multiplication.  
    - **Tasks**:
      - Write a naive matrix multiplication kernel:
        ```cpp
        __global__ void matMul(float *A, float *B, float *C, int M, int N, int K) {
            int i = blockIdx.y * blockDim.y + threadIdx.y;
            int j = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < M && j < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        ```
      - For FC1: Flatten pooling output (e.g., 14x14x64 = 12,544) to a vector, multiply by a 12,544x384 weight matrix.
      - Initialize weights randomly and test with a small input.
    - **Action**: Test FC1 output (384 units) and chain with ReLU.

#### Evening (6:30 PM - 8:30 PM): Forward Pass
12. **Step 12: Combine Layers into Forward Pass (2 hours)**  
    - **Objective**: Build a forward pass for AlexNet.  
    - **Tasks**:
      - Write a function to chain layers:
        ```cpp
        void forward(float *input, float *output, int inW, int inH, int inC) {
            float *conv1_out, *relu1_out, *pool1_out, *conv2_out, *relu2_out, *pool2_out;
            // Allocate device memory for intermediate outputs
            int outW1 = inW - 5 + 1, outH1 = inH - 5 + 1; // Conv1: 5x5 kernel
            int outC1 = 64;
            cudaMalloc(&conv1_out, outC1 * outW1 * outH1 * sizeof(float));
            cudaMalloc(&relu1_out, outC1 * outW1 * outH1 * sizeof(float));
            cudaMalloc(&pool1_out, outC1 * (outW1 / 2) * (outH1 / 2) * sizeof(float));
            // Conv1
            dim3 threads(16, 16);
            dim3 blocks((outW1 + 15) / 16, (outH1 + 15) / 16, outC1);
            conv2dMulti<<<blocks, threads>>>(input, d_conv1_weights, conv1_out, inW, inH, 5, 5, inC, outC1);
            // ReLU1
            relu<<<(outC1 * outW1 * outH1 + 255) / 256, 256>>>(conv1_out, relu1_out, outC1 * outW1 * outH1);
            // Pool1
            maxPool<<<blocks, threads>>>(relu1_out, pool1_out, outW1, outH1, 2, 2);
            // Continue for Conv2, FC layers...
            // Cleanup
            cudaFree(conv1_out); cudaFree(relu1_out); cudaFree(pool1_out);
        }
        ```
      - Test with a CIFAR-10 image and verify final output (10 units).
    - **Action**: Debug any dimension mismatches or CUDA errors.

---

### Day 3: Training and Optimization
**Goal**: Implement backpropagation, train AlexNet, and optimize performance.

#### Morning (9:00 AM - 12:30 PM): Backpropagation
13. **Step 13: Understand Backpropagation (1 hour)**  
    - **Objective**: Learn gradients for convolution, ReLU, and pooling.  
    - **Tasks**:
      - Read CS231n notes on backpropagation (~30 mins).
      - Key concepts:
        - Convolution: Compute gradients for weights and inputs.
        - ReLU: Pass gradient if input > 0.
        - Pooling: Pass gradient to max location.
        - Fully connected: Matrix multiplication gradients.
    - **Action**: Sketch gradient formulas for convolution on paper.

14. **Step 14: Convolution Gradients (2 hours)**  
    - **Objective**: Implement gradient kernels for convolution weights and inputs.  
    - **Tasks**:
      - Weight gradient kernel:
        ```cpp
        __global__ void convWeightGrad(float *input, float *gradOutput, float *gradKernel, int inW, int inH, int kW, int kH, int inC, int outC) {
            int oc = blockIdx.z; // Output channel
            int c = blockIdx.y; // Input channel
            int ki = threadIdx.y;
            int kj = threadIdx.x;
            if (oc < outC && c < inC && ki < kH && kj < kW) {
                float sum = 0.0f;
                for (int i = 0; i < inH - kH + 1; i++) {
                    for (int j = 0; j < inW - kW + 1; j++) {
                        sum += input[c * inW * inH + (i + ki) * inW + (j + kj)] *
                               gradOutput[oc * (inW - kW + 1) * (inH - kH + 1) + i * (inW - kW + 1) + j];
                    }
                }
                gradKernel[oc * inC * kW * kH + c * kW * kH + ki * kW + kj] = sum;
            }
        }
        ```
      - Input gradient kernel (similar, reverse convolution).
      - Test gradients numerically (compare with finite differences).
    - **Action**: Verify gradients for Conv1 using a small input.

#### Afternoon (1:30 PM - 5:30 PM): Training Loop
15. **Step 15: Implement Training Loop (2 hours)**  
    - **Objective**: Write a training loop with SGD.  
    - **Tasks**:
      - Compute cross-entropy loss (CPU or CUDA):
        ```cpp
        float crossEntropy(float *output, int *target, int size) {
            float loss = 0.0f;
            for (int i = 0; i < size; i++) {
                loss -= log(output[target[i]]);
            }
            return loss / size;
        }
        ```
      - SGD update kernel:
        ```cpp
        __global__ void sgdUpdate(float *weights, float *grads, float lr, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                weights[idx] -= lr * grads[idx];
            }
        }
        ```
      - Write a training loop:
        ```cpp
        for (int epoch = 0; epoch < 10; epoch++) {
            float totalLoss = 0.0f;
            for (int i = 0; i < numImages; i++) {
                forward(input[i], output, inW, inH, inC);
                totalLoss += crossEntropy(output, target[i], 10);
                backward(output, target[i]); // Implement backward pass
                sgdUpdate<<<...>>>(weights, gradWeights, 0.01f, weightSize);
            }
            printf("Epoch %d, Loss: %f\n", epoch, totalLoss / numImages);
        }
        ```
      - Train on a small batch (e.g., 100 CIFAR-10 images).
    - **Action**: Monitor loss and check if it decreases.

16. **Step 16: Test Accuracy (1 hour)**  
    - **Objective**: Evaluate model on a test set.  
    - **Tasks**:
      - Run forward pass on test images and compute accuracy:
        ```cpp
        int correct = 0;
        for (int i = 0; i < numTestImages; i++) {
            forward(testInput[i], output, inW, inH, inC);
            int pred = argmax(output, 10);
            if (pred == testTarget[i]) correct++;
        }
        printf("Accuracy: %f\n", (float)correct / numTestImages);
        ```
    - **Action**: Test on 100 test images and aim for >10% accuracy (random baseline).

#### Evening (6:30 PM - 8:30 PM): Optimization and Debugging
17. **Step 17: Optimize and Debug (2 hours)**  
    - **Objective**: Improve performance and fix issues.  
    - **Tasks**:
      - Use `nvprof` or NVIDIA Nsight to profile kernels.
      - Optimize convolution with shared memory (already introduced).
      - Experiment with block sizes (e.g., 16x16 vs. 32x32).
      - Debug: Check intermediate outputs (e.g., save Conv1 output to file).
      - Compare with CPU implementation for correctness.
    - **Action**: Measure speedup from shared memory and fix any NaN/loss issues.

---

### Notes and Tips
- **Simplifications**: Skip dropout and batch normalization to save time. Use CIFAR-10 instead of ImageNet. Avoid cuDNN unless you’re comfortable.
- **Debugging**:
  - Always check `cudaGetLastError` after kernel launches.
  - Use `cudaDeviceSynchronize` to catch errors.
  - Print intermediate outputs for small inputs to verify correctness.
- **Performance**:
  - Start with small block sizes (16x16) and adjust based on profiling.
  - If time permits, explore cuBLAS for matrix multiplication or cuDNN for convolution.
- **Resources**:
  - CUDA samples (`/usr/local/cuda/samples` or equivalent).
  - Mark Harris’s CUDA blog (NVIDIA DevBlogs).
  - PyTorch AlexNet implementation (for reference, not copying).
- **Post-Course**:
  - Study cuDNN for optimized layers.
  - Implement full AlexNet with ImageNet.
  - Explore advanced CUDA features (e.g., streams, unified memory).

---

### Daily Schedule Summary
- **Day 1**: CUDA basics, matrix addition, convolution (CPU and CUDA), project setup.
- **Day 2**: Multi-channel convolution, ReLU, pooling, fully connected layers, forward pass.
- **Day 3**: Backpropagation, training loop, optimization, and testing.

This plan is intensive but structured to build CUDA skills while working toward a simplified AlexNet. If you encounter issues, focus on debugging one layer at a time or simplify (e.g., fewer layers). Let me know if you need help with specific steps or code!