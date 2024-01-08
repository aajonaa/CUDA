#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Modify this value as needed

// CPU version for dot product calculation
double dotProductCPU(double* a, double* b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// CUDA kernel to calculate dot product using global memory for double arrays
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void dotProductGlobalMemory(double* a, double* b, double* result, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double tempResult = 0.0;

    while (tid < size) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAddDouble(result, tempResult);
}

// CUDA kernel to calculate dot product using shared memory for double arrays
__global__ void dotProductSharedMemory(double* a, double* b, double* result, int size) {
    __shared__ double temp[256]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double tempResult = 0.0;
    while (tid < size) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    temp[localIndex] = tempResult;

    // Synchronize within the block
    __syncthreads();

    // Reduction in shared memory
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (localIndex < i) {
            temp[localIndex] += temp[localIndex + i];
        }
        __syncthreads();
    }

    // Store result to global memory with atomic operation
    if (localIndex == 0) {
        atomicAddDouble(result, temp[0]);
    }
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    double *vectorA, *vectorB, *resultCPU, *resultGlobalMemory, *resultSharedMemory;
    vectorA = (double*)malloc(VECTOR_SIZE * sizeof(double));
    vectorB = (double*)malloc(VECTOR_SIZE * sizeof(double));
    resultCPU = (double*)malloc(sizeof(double)); // For CPU version
    resultGlobalMemory = (double*)malloc(sizeof(double)); // For global memory version
    resultSharedMemory = (double*)malloc(sizeof(double)); // For shared memory version

    // Check for memory allocation errors
    if (vectorA == NULL || vectorB == NULL || resultCPU == NULL || resultGlobalMemory == NULL || resultSharedMemory == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = (double)rand() / RAND_MAX; // Generating random values between 0 and 1
        vectorB[i] = (double)rand() / RAND_MAX;
    }

    // CPU version timing
    clock_t cpu_start = clock();
    *resultCPU = dotProductCPU(vectorA, vectorB, VECTOR_SIZE);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    // CUDA initialization and memory allocation
    // ... (Same memory allocations as before)

    // CUDA kernel invocation and timing for global memory version
    // ... (Same kernel invocations and timing as before)

    // CUDA kernel invocation and timing for shared memory version
    // ... (Same kernel invocations and timing as before)

    // Printing results and timing information
    printf("CPU Dot Product Result: %f\n", *resultCPU);
    printf("Global Memory Dot Product Result: %f\n", *resultGlobalMemory);
    printf("Shared Memory Dot Product Result: %f\n", *resultSharedMemory);
    printf("CPU Time: %f ms\n", cpu_time);
    printf("Global Memory Time: %f ms\n", global_time);
    printf("Shared Memory Time: %f ms\n", shared_time);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(resultCPU);
    free(resultGlobalMemory);
    free(resultSharedMemory);

    return 0;
}
