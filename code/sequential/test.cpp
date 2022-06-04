#include "sequential_sorts.h"
#include <stdio.h>
#include <omp.h>

int main() {
    int N = 16;
    
    // int arr[N] = {0, -5, 2, 8, 19, 4, 1, 2, 10, 2};
    // int arr[N] = {0};
    // int arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // int arr[N] = {10, 4, 1, 5, 2, 6, 3, 8, 9, 7};
    // float arr[N] = {0.1, 0.2, 0.3, 0.3, 0.5, -0.6, 0.7, 0.8, 0.19, 0.11};
    // int arr[N] = {7, 4, 1, 5, 2, 6, 3, 8};
    int arr[N] = {7, 4, 13, 5, 1, 6, 16, 8, 12, 9, 11, 3, 15, 2, 10, 14};
    // float arr[N] = {7.2, 4.5, 13.301, 4.7, 1.0, 6.12, 16.235, 8.99, 12.3, 9.0, 11.11, 3.0, 15.0, 2.0, 10.0, 14.0};


    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    double time = omp_get_wtime();
    seq::batcherOddEvenMergeSort(arr, N);
    time = omp_get_wtime() - time;

    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    printf("%f\n", time);
}