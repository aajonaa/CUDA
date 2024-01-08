#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Vector size

// CPU calculation
int dotProductCPU(int* a, int* b) {
    int result = 0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Global memory calculation
__global__ void dotProductGlobalMemory(int* a, int* b, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tempResult = 0;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    result[blockIdx.x * blockDim.x + threadIdx.x] = tempResult;
}

// Shared memory calculation
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

    // Synchronize threads within the block
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

    // Store the result in global memory
    if (localIndex == 0) {
        result[blockIdx.x] = temp[0];
    }
}

int main() {
    srand(time(NULL)); // Random number generation

    // Vector and result allocation
    int *vectorA, *vectorB, *result;
    vectorA = (int*)malloc(VECTOR_SIZE * sizeof(int));
    vectorB = (int*)malloc(VECTOR_SIZE * sizeof(int));
    result = (int*)malloc(256 * sizeof(int)); // 256 threads per block

    // ... (Error checking for memory allocation remains unchanged)

    // Vector initialization
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = rand() % 100; // Keep values within 100
        vectorB[i] = rand() % 100;
    }

    // CPU version timing and result printing
    clock_t cpu_start = clock();
    int cpuResult = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("CPU Dot Product Result: %d\n", cpuResult);
    printf("CPU Time: %f ms\n", cpu_time);

    // ... (GPU versions calculation and result handling remain unchanged)

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(result);

    return 0;
}
