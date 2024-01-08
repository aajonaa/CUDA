#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 100000000 // 设置向量大小

// CPU 计算
int dotProductCPU(int* a, int* b) {
    int result = 0;
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// 全局共享内存计算
__global__ void dotProductGlobalMemory(int* a, int* b, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tempResult = 0;

    while (tid < VECTOR_SIZE) {
        tempResult += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    result[blockIdx.x * blockDim.x + threadIdx.x] = tempResult;
}

// 共享内存计算
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

    // 同步块
    __syncthreads();

    // 减少共享内存
    int i = blockDim.x / 2;
    while (i != 0) {
        if (localIndex < i) {
            temp[localIndex] += temp[localIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // 存结果到全局内存
    if (localIndex == 0) {
        result[blockIdx.x] = temp[0];
    }
}

int main() {
    srand(time(NULL)); // 随机数生成

    int *vectorA, *vectorB, *result;
    vectorA = (int*)malloc(VECTOR_SIZE * sizeof(int));
    vectorB = (int*)malloc(VECTOR_SIZE * sizeof(int));
    result = (int*)malloc(256 * sizeof(int)); // 每块256个线程

    if (vectorA == NULL || vectorB == NULL || result == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // 生成向量
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vectorA[i] = rand() % 100; // 每维控制在100内
        vectorB[i] = rand() % 100;
    }

    // CPU version timing
    clock_t cpu_start = clock();
    int cpuResult = dotProductCPU(vectorA, vectorB);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    printf("CPU Dot Product Result: %d\n", cpuResult);
    printf("CPU Time: %f ms\n", cpu_time);

    // Global memory version timing
    int *dev_a, *dev_b, *dev_result;
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_result, 256 * sizeof(int));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t global_start, global_stop;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_stop);
    cudaEventRecord(global_start);

    dotProductGlobalMemory<<<256, 256>>>(dev_a, dev_b, dev_result);

    cudaMemcpy(result, dev_result, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(global_stop);
    cudaEventSynchronize(global_stop);
    float global_milliseconds = 0;
    cudaEventElapsedTime(&global_milliseconds, global_start, global_stop);

    printf("Global Memory Dot Product Result: %d\n", result[0]);
    printf("Global Memory Time: %f milliseconds\n", global_milliseconds);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    // Shared memory version timing
    cudaMalloc((void**)&dev_a, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_b, VECTOR_SIZE * sizeof(int));
    cudaMalloc((void**)&dev_result, 256 * sizeof(int));
    cudaMemcpy(dev_a, vectorA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, vectorB, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t shared_start, shared_stop;
    cudaEventCreate(&shared_start);
    cudaEventCreate(&shared_stop);
    cudaEventRecord(shared_start);

    dotProductSharedMemory<<<256, 256>>>(dev_a, dev_b, dev_result);

    cudaMemcpy(result, dev_result, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(shared_stop);
    cudaEventSynchronize(shared_stop);
    float shared_milliseconds = 0;
    cudaEventElapsedTime(&shared_milliseconds, shared_start, shared_stop);

    printf("Shared Memory Dot Product Result: %d\n", result[0]);
    printf("Shared Memory Time: %f milliseconds\n", shared_milliseconds);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);

    // Free allocated memory
    free(vectorA);
    free(vectorB);
    free(result);

    return 0;
}
