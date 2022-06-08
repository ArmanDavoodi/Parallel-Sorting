#ifndef OMP_SORT_ODD_EVEN
#define OMP_SORT_ODD_EVEN

#include "omp_sorts_headers.h"

namespace omp_par {

    template<typename Num>
    void oddEvenSort(Num* arr, int N, double& deltaTime) {
        double t = omp_get_wtime();

        for (int phase = 0; phase < N; ++phase) {
            #pragma omp parallel for num_threads(P)
            for (int i = phase % 2; i < (N - 1); i += 2) {
                if (arr[i] > arr[i+1])
                    std::swap(arr[i], arr[i+1]);
            }
        }

        deltaTime += omp_get_wtime() - t;
    }

}

#endif