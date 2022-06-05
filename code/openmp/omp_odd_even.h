#ifndef OMP_SORT_ODD_EVEN
#define OMP_SORT_ODD_EVEN

#include "omp_sorts_headers.h"

namespace omp_par {

    template<typename Num>
    void oddEvenSort(Num* arr, int N) {
        int sorted = false, start = 1;
        #pragma omp parallel num_threads(P)
        {
            for (int phase = 0; phase < N; ++phase) {
                #pragma omp for
                for (int i = phase % 2; i < (N - 1); i += 2) {
                    if (arr[i] > arr[i+1]) {
                        sorted = false;
                        std::swap(arr[i], arr[i+1]);
                    }
                }
            }
        }
    }

}

#endif