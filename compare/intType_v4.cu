#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000

// CPU dot product calculation
int dotProductCPU(int* a, int* b) {
    int result = 0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Global memory dot product calculation
__global__ void dotProductGlobalMemory(int* a, int* b, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tempResult = 0;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(result, tempResult); // Add partial results atomically
}

// Shared memory dot product calculation
__global__ void dotProductSharedMemory(int* a, int* b, int* result) {
    __shared__ int temp[256]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    int tempResult = 0;
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

    // Vector allocation and initialization
    int *vectorA, *vectorB, *resultCPU, *resultGlobal, *resultShared;
    vectorA = (int*)malloc(VECTOR_SIZE * sizeof(int));
    vectorB = (int*)malloc(VECTOR_SIZE * sizeof(int));
    resultCPU = (int*)malloc(sizeof(int));
    resultGlobal = (int*)malloc(sizeof(int));
    resultShared = (int*)malloc(sizeof(int));

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = rand() % 100; // Keeping values within 100
        vectorB[i] = rand() % 100;
    }

    // CPU dot product calculation and timing
    clock_t cpu_start = clock();
    *resultCPU = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("CPU Dot Product Result: %d\n", *resultCPU);
    printf("CPU Time: %f ms\n", cpu_time);

    // GPU (Global Memory) dot product calculation and timing
    int *dev_a, *dev_b, *dev_resultGlobal;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_resultGlobal, sizeof(int));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_resultGlobal, 0, sizeof(int)); // Initialize result to 0

    clock_t global_start = clock();
    dotProductGlobalMemory<<<256, 256>>>(dev_a, dev_b, dev_resultGlobal);
    cudaDeviceSynchronize();
    cudaMemcpy(resultGlobal, dev_resultGlobal, sizeof(int), cudaMemcpyDeviceToHost);
    clock_t global_end = clock();
    double global_time = ((double)(global_end - global_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("Global Memory Dot Product Result: %d\n", *resultGlobal);
    printf("Global Memory Time: %f ms\n", global_time);

    // GPU (Shared Memory) dot product calculation and timing
    int *dev_resultShared;
    cudaMalloc((void**)&dev_resultShared, sizeof(int));
    cudaMemset(dev_resultShared, 0, sizeof(int)); // Initialize result to 0

    clock_t shared_start = clock();
    dotProductSharedMemory<<<256, 256>>>(dev_a, dev_b, dev_resultShared);
    cudaDeviceSynchronize();
    cudaMemcpy(resultShared, dev_resultShared, sizeof(int), cudaMemcpyDeviceToHost);
    clock_t shared_end = clock();
    double shared_time = ((double)(shared_end - shared_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("Shared Memory Dot Product Result: %d\n", *resultShared);
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
