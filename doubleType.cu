#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CPU dot product calculation for double arrays
double dotProductCPU(double* a, double* b) {
    double result = 0.0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Global memory dot product calculation for double arrays
__global__ void dotProductDoubleGlobalMemory(double* a, double* b, double* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double tempResult = 0.0;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    atomicAddDouble(result, tempResult);
}

int main() {
    srand(time(NULL));

    // Vector allocation and initialization for double arrays
    double *vectorA, *vectorB, *resultCPU, *resultGlobal;
    vectorA = (double*)malloc(VECTOR_SIZE * sizeof(double));
    vectorB = (double*)malloc(VECTOR_SIZE * sizeof(double));
    resultCPU = (double*)malloc(sizeof(double));
    resultGlobal = (double*)malloc(sizeof(double));

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = static_cast<double>(rand()) / RAND_MAX; // Random values between 0 and 1
        vectorB[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // CPU dot product calculation and timing
    clock_t cpu_start = clock();
    *resultCPU = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("CPU Dot Product Result: %f\n", *resultCPU);
    printf("CPU Time: %f ms\n", cpu_time);

    // GPU (Global Memory) dot product calculation and timing
    double *dev_a, *dev_b, *dev_resultGlobal;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_resultGlobal, sizeof(double));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dev_resultGlobal, 0, sizeof(double)); // Initialize result to 0

    clock_t global_start = clock();
    dotProductDoubleGlobalMemory<<<256, 256>>>(dev_a, dev_b, dev_resultGlobal);
    cudaDeviceSynchronize();
    cudaMemcpy(resultGlobal, dev_resultGlobal, sizeof(double), cudaMemcpyDeviceToHost);
    clock_t global_end = clock();
    double global_time = ((double)(global_end - global_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("Global Memory Dot Product Result: %f\n", *resultGlobal);
    printf("Global Memory Time: %f ms\n", global_time);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(resultCPU);
    free(resultGlobal);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_resultGlobal);

    return 0;
}
