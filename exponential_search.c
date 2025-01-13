#include <stdio.h>
#include <stdlib.h>

// Function to perform exponential search
int exponentialSearch(int arr[], int n, int x) {
    if (arr[0] == x)
        return 0;

    int i = 1;
    while (i < n && arr[i] <= x)
        i *= 2;

    // Perform binary search in the found range
    int low = i / 2;
    int high = i < n ? i : n - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] == x)
            return mid; // Element found
        else if (arr[mid] < x)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1; // Element not found
}

int main() {
    // Open the file
    FILE *file = fopen("random_numbers.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Count the number of elements in the file
    int n = 0;
    int temp;
    while (fscanf(file, "%d", &temp) != EOF) {
        n++;
    }

    // Move back to the beginning of the file
    rewind(file);

    // Allocate memory for the array
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return 1;
    }

    // Read the array elements from the file
    for (int i = 0; i < n; i++) {
        fscanf(file, "%d", &arr[i]);
    }

    // Close the file
    fclose(file);

    // Element to search for
    int target = 5000;

    // Perform exponential search
    int result = exponentialSearch(arr, n, target);

    // Display the result
    if (result != -1) {
        printf("Element %d found at index %d\n", target, result);
    } else {
        printf("Element %d not found in the array\n", target);
    }

    // Free allocated memory
    free(arr);

    return 0;
}
