#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int exponentialSearch(int arr[], int start, int end, int x, int rank,int localSize) {
 //printf("start=%d end = %d\n",start,end);
 //for(int i=start;i<end;i++)
 //{
  //  printf("\n %d",arr[i]);
 //}
  if (arr[start] == x)
        return start;

    int i = start + 1;
    while (i <= end && arr[i] <= x)
        i *= 2;

    // Perform binary search in the found range
    int low = i / 2;
    int high = (i <= end) ? i : end;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] == x)
           {//if (rank!=0)
                //return rank-1*localSize+mid;
            //else
                return  rank*localSize+mid; // Element found
           }
        else if (arr[mid] < x)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1; // Element not found
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
     
      int maxArraySize = 15000; 
      //int array[maxArraySize];
    int totalSize, localSize;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime, endTime, searchTime;
    int* localArr = NULL;
    int* globalArr = NULL;
    localArr = (int *)malloc(maxArraySize * sizeof(int));
    int *  array = (int *)malloc(maxArraySize * sizeof(int));
        globalArr = (int *)malloc(maxArraySize * sizeof(int));

    if (rank == 0) {
        // Open the file
        FILE *file = fopen("random_numbers.txt", "r");
        if (file == NULL) {
            perror("Error opening file");
            MPI_Finalize();
            return 1;
        }

          // Define the maximum array size
   
    int arraySize = 0; // Initialize the array size

    // Read integers from the file and store them in the array
    while (fscanf(file, "%d", &array[arraySize]) != EOF) {
        arraySize++;
        if (arraySize >= maxArraySize) {
            printf("Array size limit reached. Increase maxArraySize.\n");
            break;
        }
    }

       localSize=arraySize/size;
       //printf(" size = %d",localSize);
        
        fclose(file);
    }

    // Broadcast the total size to all processes
   MPI_Bcast(&localSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

 

    // Scatter the entire array to each process
    MPI_Scatter(&array[0], localSize, MPI_INT, &globalArr[0], localSize, MPI_INT, 0, MPI_COMM_WORLD);
    //if (rank==2)
    //{
   
    //}
    // Element to search for
    int target = 4575;

    // Start measuring time for parallel search
    startTime = MPI_Wtime();

    // Perform exponential search in parallel
    int localResult = exponentialSearch(globalArr, 0 , localSize, target, rank,localSize);

    // Stop measuring time for parallel search
    endTime = MPI_Wtime();

    // Collect results from all processes
    int globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // Display the result and time on process 0
    if (rank == 0) {
        searchTime = endTime - startTime;
        if (globalResult != -1) {
            printf("Element %d found at index %d\n", target, globalResult);
        } else {
            printf("Element %d not found in the array\n", target);
        }

        printf("Time taken for the parallel search: %f seconds\n", searchTime);
    }

    // Clean up memory
    free(localArr);
    free(globalArr);
    free(array);
    MPI_Finalize();

    return 0;
}
