#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Change this value to modify the vector size

// CUDA kernel to calculate dot product using shared memory
__global__ void dotProductCUDA(double* a, double* b, double* result) {
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
        result[blockIdx.x] = temp[0];
    }
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    double *vectorA, *vectorB, *result;
    vectorA = (double*)malloc(VECTOR_SIZE * sizeof(double));
    vectorB = (double*)malloc(VECTOR_SIZE * sizeof(double));
    result = (double*)malloc(256 * sizeof(double)); // 256 threads per block

    if (vectorA == NULL || vectorB == NULL || result == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = (double)rand() / RAND_MAX; // Generating random values between 0 and 1
        vectorB[i] = (double)rand() / RAND_MAX;
    }

    double *dev_a, *dev_b, *dev_result;

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&dev_result, 256 * sizeof(double));

    // Copy input arrays from host to device
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    dotProductCUDA<<<256, 256>>>(dev_a, dev_b, dev_result);

    // Copy result array from device to host
    cudaMemcpy(result, dev_result, 256 * sizeof(double), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    double finalResult = 0.0;
    for (int i = 0; i < 256; ++i) {
        finalResult += result[i];
    }

    // Measure and print the execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Dot product: %f\n", finalResult);
    printf("Execution time: %f milliseconds\n", milliseconds);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(result);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    return 0;
}
