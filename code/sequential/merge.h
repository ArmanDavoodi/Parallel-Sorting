#ifndef SORT_MERGE
#define SORT_MERGE

#include "sorts_headers.h"
#include "insertion.h"

namespace seq {
    inline int getMinRunSize(int N) {
        int r = 0;
        while (N >= 64) {
            r |= N & 1;
            N >>= 1;
        }
        return N + r;
    }

    // merge two blocks of sorted arrays -> midx is the starting index of the right array
    template<typename Num>
    inline void merge(Num* arr, int lidx, int midx, int ridx) {
        int nl = midx - lidx, nr = ridx - midx + 1;
        Num* left = new Num[nl];
        Num* right = new Num[nr];
        register int i, j, k;

        // copy the two blocks into two temp arrays
        for (i = 0; i < nl; ++i)
            left[i] = arr[i + lidx];
        for (i = 0; i < nr; ++i)
            right[i] = arr[i + midx];
        
        // merge algorithm
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

        // put the remaining elements in the array
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

        delete[] left;
        delete[] right;

    }

    // bottom-up implementation
    template<typename Num>
    void mergeSort(Num* arr, int N) {        
        register int currentSize = getMinRunSize(N), i;

        // ad-hoc -> sort smaller blocks using insertion sort
        if (currentSize > 1) {
            for (i = 0; i < N; i += currentSize)
                seq::insertionSort(arr + i, i + currentSize < N ? currentSize : N - i);
        }
        
        for (; currentSize < N; currentSize *= 2) {
            for (i = 0; i < N - 1; i += 2*currentSize) {
                int mid = std::min(i + currentSize, N-1);
                int right = std::min(i + 2*currentSize - 1, N-1);

                if (arr[mid - 1] > arr[mid]) // merge two blocks if they are not already sorted
                    merge(arr, i, mid, right);
            }
        }
    }

}

#endif