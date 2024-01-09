#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000

// CPU dot product calculation for float arrays
float dotProductCPU(float* a, float* b) {
    float result = 0.0f;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Global memory dot product calculation for float arrays
__global__ void dotProductGlobalMemory(float* a, float* b, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float tempResult = 0.0f;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(result, tempResult); // Add partial results atomically
}

// Shared memory dot product calculation for float arrays
__global__ void dotProductSharedMemory(float* a, float* b, float* result) {
    __shared__ float temp[256]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    float tempResult = 0.0f;
    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    temp[localIndex] = tempResult;

    __syncthreads(); // Synchronize threads within the block

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (localIndex < i) {
            temp[localIndex] += temp[localIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store the result in global memory
    if (localIndex == 0) {
        atomicAdd(result, temp[0]); // Add the block's result atomically
    }
}

int main() {
    srand(time(NULL));

    // Vector allocation and initialization for float arrays
    float *vectorA, *vectorB, *resultCPU, *resultGlobal, *resultShared;
    vectorA = (float*)malloc(VECTOR_SIZE * sizeof(float));
    vectorB = (float*)malloc(VECTOR_SIZE * sizeof(float));
    resultCPU = (float*)malloc(sizeof(float));
    resultGlobal = (float*)malloc(sizeof(float));
    resultShared = (float*)malloc(sizeof(float));

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = static_cast<float>(rand()) / RAND_MAX; // Random values between 0 and 1
        vectorB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU dot product calculation and timing
    clock_t cpu_start = clock();
    *resultCPU = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("CPU Dot Product Result: %f\n", *resultCPU);
    printf("CPU Time: %f ms\n", cpu_time);

    // GPU (Global Memory) dot product calculation and timing
    float *dev_a, *dev_b, *dev_resultGlobal;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_resultGlobal, sizeof(float));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dev_resultGlobal, 0, sizeof(float)); // Initialize result to 0

    clock_t global_start = clock();
    dotProductGlobalMemory<<<256, 256>>>(dev_a, dev_b, dev_resultGlobal);
    cudaDeviceSynchronize();
    cudaMemcpy(resultGlobal, dev_resultGlobal, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t global_end = clock();
    double global_time = ((double)(global_end - global_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("Global Memory Dot Product Result: %f\n", *resultGlobal);
    printf("Global Memory Time: %f ms\n", global_time);

    // GPU (Shared Memory) dot product calculation and timing
    float *dev_resultShared;
    cudaMalloc((void**)&dev_resultShared, sizeof(float));
    cudaMemset(dev_resultShared, 0, sizeof(float)); // Initialize result to 0

    clock_t shared_start = clock();
    dotProductSharedMemory<<<256, 256>>>(dev_a, dev_b, dev_resultShared);
    cudaDeviceSynchronize();
    cudaMemcpy(resultShared, dev_resultShared, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t shared_end = clock();
    double shared_time = ((double)(shared_end - shared_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("Shared Memory Dot Product Result: %f\n", *resultShared);
    printf("Shared Memory Time: %f ms\n", shared_time);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(resultCPU);
    free(resultGlobal);
    free(resultShared);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resultGlobal);
    cudaFree(dev_resultShared);

    return 0;
}
