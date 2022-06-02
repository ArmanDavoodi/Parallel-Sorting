#ifndef SORT_MERGE
#define SORT_MERGE

#include "sorts_headers.h"
#include "insertion.h"

namespace seq {
    constexpr int MERGE_SORT_AD_HOC = 1; // todo

    template<typename Num>
    inline void merge(Num* arr, int lidx, int midx, int ridx) {
        int nl = midx - lidx, nr = ridx - midx + 1;
        Num* left = new Num[nl];
        Num* right = new Num[nr];
        register int i, j, k;

        for (i = 0; i < nl; ++i)
            left[i] = arr[i + lidx];
        for (i = 0; i < nr; ++i)
            right[i] = arr[i + midx];
        
        i = 0;
        j = 0;
        k = lidx;
        while (i < nl && j < nr) {
            if (left[i] <= right[j]) {
                arr[k] = left[i];
                ++i;
            }
            else {
                arr[k] = right[j];
                ++j;
            }
            ++k;
        }

        while (i < nl) {
            arr[k] = left[i];
            ++i;
            ++k;
        }

        while (j < nr) {
            arr[k] = right[j];
            ++j;
            ++k;
        }

    }

    template<typename Num>
    double mergeSort(Num* arr, int N) {
        double start = omp_get_wtime();
        
        register int currentSize = MERGE_SORT_AD_HOC, i;

        if (currentSize > 1) {
            for (i = 0; i < N; i += currentSize)
                seq::insertionSort(arr + i, i + currentSize < N ? currentSize : N - i);
        }
        
        for (; currentSize < N; currentSize *= 2) {
            for (i = 0; i < N - 1; i += 2*currentSize) {
                int mid = std::min(i + currentSize, N-1);
                int right = std::min(i + 2*currentSize - 1, N - 1);

                if (arr[mid - 1] > arr[mid])
                    merge(arr, i, mid, right);
            }
        }

        return omp_get_wtime() - start;
    }

}

#endif