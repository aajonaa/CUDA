#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Change this value to modify the vector size

// CUDA kernel to calculate dot product using global memory
__global__ void dotProductCUDA(float* a, float* b, float* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float tempResult = 0.0f;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    result[blockIdx.x * blockDim.x + threadIdx.x] = tempResult;
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    float *vectorA, *vectorB, *result;
    vectorA = (float*)malloc(VECTOR_SIZE * sizeof(float));
    vectorB = (float*)malloc(VECTOR_SIZE * sizeof(float));
    result = (float*)malloc(256 * sizeof(float)); // 256 threads per block

    if (vectorA == NULL || vectorB == NULL || result == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = (float)rand() / RAND_MAX; // Generating random values between 0 and 1
        vectorB[i] = (float)rand() / RAND_MAX;
    }

    float *dev_a, *dev_b, *dev_result;

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&dev_result, 256 * sizeof(float));

    // Copy input arrays from host to device
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    dotProductCUDA<<<256, 256>>>(dev_a, dev_b, dev_result);

    // Copy result array from device to host
    cudaMemcpy(result, dev_result, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    float finalResult = 0.0f;
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
