#ifndef SORT_INSERTION
#define SORT_INSERTION

#include "sorts_headers.h"

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

#endif