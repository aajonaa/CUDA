#include <stdio.h>

#define N 10

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x; // Calculate the thread's global ID
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N]; // Host arrays
    int *dev_a, *dev_b, *dev_c; // Device arrays

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // Copy input arrays from host to device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    add<<<N, 1>>>(dev_a, dev_b, dev_c);

    // Copy result array from device to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
