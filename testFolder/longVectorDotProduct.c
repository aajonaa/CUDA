#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 1000000 // Change this value to modify the vector size

// Function to generate random values for the vectors
void generateRandomVector(double* vector, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] = (double)rand() / RAND_MAX; // Generating random values between 0 and 1
    }
}

// Function to calculate the dot product of two vectors
double dotProduct(double* vectorA, double* vectorB, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += vectorA[i] * vectorB[i];
    }
    return result;
}

int main() {
    srand(time(NULL)); // Seed for random number generation

    double *vectorA, *vectorB;
    vectorA = (double*)malloc(VECTOR_SIZE * sizeof(double));
    vectorB = (double*)malloc(VECTOR_SIZE * sizeof(double));

    if (vectorA == NULL || vectorB == NULL) {
        printf("Memory allocation failed.\n");
        return -1;
    }

    // Generating random values for the vectors
    generateRandomVector(vectorA, VECTOR_SIZE);
    generateRandomVector(vectorB, VECTOR_SIZE);

    // Calculating the dot product
    clock_t start_time = clock();
    double result = dotProduct(vectorA, vectorB, VECTOR_SIZE);
    clock_t end_time = clock();

    // Printing the result and execution time
    printf("Dot product: %f\n", result);
    printf("Execution time: %f seconds\n", ((double)(end_time - start_time)) / CLOCKS_PER_SEC);

    // Free allocated memory
    free(vectorA);
    free(vectorB);

    return 0;
}
