#ifndef SORT_BUBBLE
#define SORT_BUBBLE

#include "sorts_headers.h"

namespace seq {

    template<typename Num>
    void bubbleSort(Num* arr, int N) {
        int sorted = false;
        for (int i = 0; !sorted && i < N - 1; ++i) {
            sorted = true;
            for (int j = 0; j < (N - i - 1); ++j) {
                if (arr[j] > arr[j+1]) {
                    sorted = false;
                    std::swap(arr[j], arr[j+1]);
                }
            }
        }
    }

}

#endif