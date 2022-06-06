#ifndef SORT_ODD_EVEN
#define SORT_ODD_EVEN

#include "sorts_headers.h"

namespace seq {

    template<typename Num>
    void oddEvenSort(Num* arr, int N, double& deltaTime) {
        double t = omp_get_wtime();

        int sorted = false;
        while (!sorted) {
            sorted = true;
            for (int i = 1; i < (N - 1); i += 2) {
                if (arr[i] > arr[i+1]) {
                    sorted = false;
                    std::swap(arr[i], arr[i+1]);
                }
            }
            for (int i = 0; i < (N - 1); i += 2) {
                if (arr[i] > arr[i+1]) {
                    sorted = false;
                    std::swap(arr[i], arr[i+1]);
                }
            }
        }

        deltaTime += omp_get_wtime() - t;
    }

}

#endif