#include "cuda_sorts.h"
#include <stdio.h>

int main() {
    int N = 10;
    
    int arr[N] = {0, -5, 2, 8, 19, 4, 1, 2, 10, 2};
    // int arr[N] = {0};
    // int arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // int arr[N] = {10, 4, 1, 5, 2, 6, 3, 8, 9, 7};
    // float arr[N] = {0.1, 0.2, 0.3, 0.3, 0.5, -0.6, 0.7, 0.8, 0.19, 0.11};
    // int arr[N] = {7, 4, 1, 5, 2, 6, 3, 8};
    // int arr[N] = {7, 4, 13, 5, 1, 6, 16, 8, 12, 9, 11, 3, 15, 2, 10, 14};
    // float arr[N] = {7.2, 4.5, 13.301, 4.7, 1.0, 6.12, 16.235, 8.99, 2.3, 9.0, 11.11, 3.0, 15.0, 2.0, 10.0, 14.0};
    // int arr[N] = {19, 12, 13, 1, 2, 2, 4, 3, 5, 8, 9, 7, 10, 15, 11, 6, 14, 16, 17, 18, 2}; 


    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    double time = 0;
    cuda_par::oddEvenSort(arr, N, time);

    for (int i = 0; i < N; ++i)
        printf("%d ", arr[i]);
    printf("\n");

    printf("%f\n", time);
}