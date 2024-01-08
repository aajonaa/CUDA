#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Change this value to modify the vector size

// CPU version for dot product calculation
float dotProductCPU(float* a, float* b) {
    float result = 0.0f;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// CUDA kernel to calculate dot product using global memory
__global__ void dotProductGlobalMemory(float* a, float* b, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float tempResult = 0.0f;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAdd(result, tempResult);
}

// CUDA kernel to calculate dot product using shared memory
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

    // Synchronize within the block
    __syncthreads();

    // Reduction in shared memory
    int i = blockDim.x / 2;
    while (i != 0) {
        if (localIndex < i) {
            temp[localIndex] += temp[localIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Store result to global memory
    if (localIndex == 0) {
        atomicAdd(result, temp[0]);
    }
}



int main() {
    srand(time(NULL)); // Seed for random number generation

    float *vectorA, *vectorB, *resultCPU, *resultGlobalMemory, *resultSharedMemory;
    vectorA = (float*)malloc(VECTOR_SIZE * sizeof(float));
    vectorB = (float*)malloc(VECTOR_SIZE * sizeof(float));
    resultCPU = (float*)malloc(sizeof(float)); // For CPU version
    resultGlobalMemory = (float*)malloc(sizeof(float)); // For global memory version
    resultSharedMemory = (float*)malloc(sizeof(float)); // For shared memory version

    if (vectorA == NULL || vectorB == NULL || resultCPU == NULL || resultGlobalMemory == NULL || resultSharedMemory == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = (float)rand() / RAND_MAX; // Generating random values between 0 and 1
        vectorB[i] = (float)rand() / RAND_MAX;
    }

    // CPU version timing
    clock_t cpu_start = clock();
    *resultCPU = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    // Global memory version timing
    float *dev_a, *dev_b, *dev_result;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    clock_t global_start = clock();
    dotProductGlobalMemory<<<1, 1>>>(dev_a, dev_b, dev_result);
    cudaMemcpy(resultGlobalMemory, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t global_end = clock();
    double global_time = ((double)(global_end - global_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    // Shared memory version timing
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    clock_t shared_start = clock();
    dotProductSharedMemory<<<1, 1>>>(dev_a, dev_b, dev_result);
    cudaMemcpy(resultSharedMemory, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t shared_end = clock();
    double shared_time = ((double)(shared_end - shared_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    // Print timing results
    printf("CPU version time: %f milliseconds\n", cpu_time);
    printf("Global memory version time: %f milliseconds\n", global_time);
    printf("Shared memory version time: %f milliseconds\n", shared_time);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(resultCPU);
    free(resultGlobalMemory);
    free(resultSharedMemory);

    return 0;
}
