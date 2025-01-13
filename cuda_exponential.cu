#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int binarySearch(int *arr, int l, int r, int x) {
    while (r >= l) {
        int mid = (l + r) / 2;
        if (arr[mid] == x)
            return mid;  // Element found
        if (arr[mid] > x)
            r = mid - 1;
        else
            l = mid + 1;
    }
    return -1;  // Element not found
}

__global__ void parallelExponentialSearch(int *arr, int size, int target, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Step 1: Exponential search
        int lowerBound = 1;
        while (lowerBound < size && arr[lowerBound] <= target) {
            lowerBound *= 2;
        }
        int upperBound = (lowerBound < size) ? lowerBound : size - 1;
        lowerBound /= 2;

        // Step 2: Binary search within the range
        int localRes = binarySearch(arr, lowerBound, upperBound, target);

        // Step 3: Use atomicMin to ensure proper synchronization
        if (localRes != -1) {
            atomicMin(result, localRes);
        }
    }
}

int main(int argc, char const *argv[]) {
    int *d_array, *d_result;
    int target = 8000;
    int result = 100000; // Initialize with a large value (not -1)

    // Read array from file
    FILE *file = fopen("random_numbers.txt", "r");
    if (file == NULL) {
        printf("Failed to open the file for reading.\n");
        return 1;
    }

    int A[10000];
    int num_elements_A = 0;
    int val;

    while (fscanf(file, "%d", &val) != EOF) {
        A[num_elements_A] = val;
        num_elements_A++;
    }

    fclose(file);

    // Allocate memory on the device
    cudaMalloc((void**)&d_array, num_elements_A * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_array, A, num_elements_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256; // Number of threads per block
    int gridSize = (num_elements_A + blockSize - 1) / blockSize;

    // Start CUDA event timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch the parallel exponential search kernel
    parallelExponentialSearch<<<gridSize, blockSize>>>(d_array, num_elements_A, target, d_result);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy the result back from device to host
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Time taken by the kernel
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Check if the target element was found
    if (result != 100000) {
        printf("Element %d found at position: %d\n", target, result);
    } else {
        printf("Element %d not found in the array\n", target);
    }

    printf("Time taken by the kernel: %f ms\n", elapsedTime);

    // Free allocated memory
    cudaFree(d_array);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
