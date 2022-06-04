#ifndef SORT_BOE_MERGE
#define SORT_BOE_MERGE

#include "sorts_headers.h"

namespace seq {
    // only works on arrays of size 2^k
    template<typename Num>
    void batcherOddEvenMergeSort(Num* arr, int N) {        
        for (int p = 1; p < N; p = 2*p) {
            for (int k = p; k > 0; k /= 2) {
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