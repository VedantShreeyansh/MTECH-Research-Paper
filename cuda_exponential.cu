#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int binarySearch(int *arr, int l, int r, int x) {
    if (r >= l) {
        int mid = (l + r )/2;

        if (arr[mid] == x)
            return mid;

        if (arr[mid] > x)
            return binarySearch(arr, l, mid-1, x);

        return binarySearch(arr, mid+1, r, x);
    }

    return -1;
}

__global__ void parallelExponentialSearch(int *arr, int size, int target, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int localRes = -1;
    if(tid < size) {
        int lowerBound = 1;
        while (lowerBound < target && arr[lowerBound] <= target) {
            lowerBound = lowerBound*2;
        }

        int upperBound = lowerBound < size-1 ? lowerBound : size-1;
        //printf("\nMin Val : %d\n", minVal);
        localRes =  binarySearch(arr, lowerBound/2, upperBound, target);
        *result = localRes;
    }
}

int main(int argc, char const *argv[]) {
    int *d_array, *d_result;
    int target = 118;
    int result;


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

    cudaMalloc((void**)&d_array, num_elements_A * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    cudaMemcpy(d_array, A, num_elements_A * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_elements_A + blockSize - 1) / blockSize;
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    parallelExponentialSearch<<<gridSize, blockSize>>>(d_array, num_elements_A, target, d_result);

    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);

    if (result != -1) {
        printf("Element %d found at postion : %d\n", target, result);
    } else {
        printf("Element %d not found in the array\n", target);
    }

    float eTMs;
    cudaEventElapsedTime(&eTMs, start, stop);
    printf("\nTime Taken by the Kernal is : %f\n", eTMs);

    cudaFree(d_array);
    cudaFree(d_result);
    fclose(file);
    return 0;
}
