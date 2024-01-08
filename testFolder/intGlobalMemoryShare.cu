#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // Change this value to modify the vector size

// CUDA kernel to calculate dot product using global memory
__global__ void dotProductCUDA(int* a, int* b, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tempResult = 0;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    result[blockIdx.x * blockDim.x + threadIdx.x] = tempResult;
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    int *vectorA, *vectorB, *result;
    vectorA = (int*)malloc(VECTOR_SIZE * sizeof(int));
    vectorB = (int*)malloc(VECTOR_SIZE * sizeof(int));
    result = (int*)malloc(256 * sizeof(int)); // 256 threads per block

    if (vectorA == NULL || vectorB == NULL || result == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = rand() % 100; // Generating random integer values (0 to 99)
        vectorB[i] = rand() % 100;
    }

    int *dev_a, *dev_b, *dev_result;

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_result, 256 * sizeof(int));

    // Copy input arrays from host to device
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    dotProductCUDA<<<256, 256>>>(dev_a, dev_b, dev_result);

    // Copy result array from device to host
    cudaMemcpy(result, dev_result, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Final reduction on CPU
    int finalResult = 0;
    for (int i = 0; i < 256; ++i) {
        finalResult += result[i];
    }

    // Measure and print the execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Dot product: %d\n", finalResult);
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
