#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Change this value to modify the vector size

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


// CPU version for dot product calculation
double dotProductCPU(double* a, double* b) {
    double result = 0.0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

__global__ void dotProductGlobalMemory(double* a, double* b, double* result) {
    __shared__ double temp[256]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double tempResult = 0.0;
    while (tid < VECTOR_SIZE) {
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
        atomicAdd(result, temp[0]);
    }
}


__global__ void dotProductSharedMemory(double* a, double* b, double* result) {
    __shared__ double temp[256]; // Shared memory for each block
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int localIndex = threadIdx.x;

    double tempResult = 0.0;
    while (tid < VECTOR_SIZE) {
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
        atomicAdd(result, temp[0]);
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
    *resultCPU = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    // Global memory version timing
    double *dev_a, *dev_b, *dev_resultGlobalMemory;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_resultGlobalMemory, sizeof(double));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (VECTOR_SIZE + blockSize - 1) / blockSize;

    clock_t global_start = clock();
    dotProductGlobalMemory<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_resultGlobalMemory);
    cudaDeviceSynchronize();
    cudaMemcpy(resultGlobalMemory, dev_resultGlobalMemory, sizeof(double), cudaMemcpyDeviceToHost);
    clock_t global_end = clock();
    double global_time = ((double)(global_end - global_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resultGlobalMemory);

    // Shared memory version timing
    double *dev_resultSharedMemory;
    cudaMalloc((void**)&dev_resultSharedMemory, sizeof(double));
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(double));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    clock_t shared_start = clock();
    dotProductSharedMemory<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_resultSharedMemory);
    cudaDeviceSynchronize();
    cudaMemcpy(resultSharedMemory, dev_resultSharedMemory, sizeof(double), cudaMemcpyDeviceToHost);
    clock_t shared_end = clock();
    double shared_time = ((double)(shared_end - shared_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resultSharedMemory);


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
