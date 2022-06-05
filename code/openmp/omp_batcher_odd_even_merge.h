#ifndef OMP_SORT_BOE_MERGE
#define OMP_SORT_BOE_MERGE

#include "omp_sorts_headers.h"

namespace omp_par {
    // only works on arrays of size 2^k
    template<typename Num>
    void batcherOddEvenMergeSort(Num* arr, int N) {    
        #pragma omp parallel num_threads(P) // use or not TODO ###############################    
        for (int p = 1; p < N; p = 2*p) {
            for (int k = p; k > 0; k /= 2) {
                #pragma omp for // TODO where to use + scheduling ##############################
                for (int j = k % p; j < N - k; j += 2*k) {
                    for (int i = 0; i < k; ++i) {
                        if ((i + j) / (2 * p) == (i + j + k) / (2 * p))
                            if (arr[i + j] > arr[i + j + k])
                                std::swap(arr[i + j], arr[i + j + k]);
                    }
                }
            }
        }
    }

}

#endif