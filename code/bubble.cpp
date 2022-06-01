#ifndef OMP_DEF
#define OMP_DEF
#include <omp.h>
#endif

// #include <stdio.h>

namespace seq {

    template<typename Num>
    double bublleSort(Num* arr, int N) {
        double start = omp_get_wtime();
        for (int i = 0; i < N - 1; ++i) {
            int sorted = true;
            for (int j = 0; j < (N - i - 1); ++j) {
                if (arr[j] > arr[j+1]) {
                    sorted = false;
                    arr[j] += arr[j+1];
                    arr[j+1] = arr[j] - arr[j+1];
                    arr[j] -= arr[j+1];
                    // Num temp = arr[j];
                    // arr[j] = arr[j+1];
                    // arr[j+1] = temp;
                }
            }
            if (sorted)
                break;
        }
        return omp_get_wtime() - start;
    }

}

// int main() {
//     int N = 10;
    
//     int arr[N] = {0, -5, 2, 8, 19, 4, 1, 2, 10, 2};
//     // int arr[N] = {0};
//     // int arr[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//     // float arr[N] = {0.1, 0.2, 0.3, 0.3, 0.5, -0.6, 0.7, 0.8, 0.19, 0.11};

//     for (int i = 0; i < N; ++i)
//         printf("%d ", arr[i]);
//     printf("\n");

//     double time = seq::bublleSort(arr, N);

//     for (int i = 0; i < N; ++i)
//         printf("%d ", arr[i]);
//     printf("\n");

//     printf("%f\n", time);
// }