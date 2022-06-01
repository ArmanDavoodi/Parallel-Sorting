#ifndef OMP_DEF
#define OMP_DEF
#include <omp.h>
#endif

// #include <stdio.h>

namespace seq {

    template<typename Num>
    double insertionSort(Num* arr, int N) {
        double start = omp_get_wtime();
        for (int i = 1; i < N; ++i) {
            int j;
            Num temp = arr[i];
            for (j = i; (j > 0) && (temp < arr[j-1]); --j) {
                arr[j] = arr[j-1];
            }
            arr[j] = temp;
        }
        return omp_get_wtime() - start;
    }

}

// int main() {
//     int N = 10;
    
//     // int arr[N] = {0, -5, 2, 8, 19, 4, 1, 2, 10, 2};
//     // int arr[N] = {0};
//     // int arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//     float arr[N] = {0.1, 0.2, 0.3, 0.3, 0.5, -0.6, 0.7, 0.8, 0.19, 0.11};

//     for (int i = 0; i < N; ++i)
//         printf("%f ", arr[i]);
//     printf("\n");

//     double time = seq::insertionSort(arr, N);

//     for (int i = 0; i < N; ++i)
//         printf("%f ", arr[i]);
//     printf("\n");

//     printf("%f\n", time);
// }