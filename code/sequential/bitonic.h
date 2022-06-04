#ifndef SORT_BITONIC
#define SORT_BITONIC

#include "sorts_headers.h"

namespace seq {
    // only works on arrays of size 2^k
    template<typename Num>
    void bitonicSort(Num* arr, int N) {        
        // size of the runs being sorted
        for (int runSize = 2; runSize <= N; runSize = 2*runSize) {
            for (int stage = runSize / 2; stage > 0; stage /= 2) {
                for (int wireLowerEnd = 0; wireLowerEnd < N; ++wireLowerEnd) {
                    int wireUpperEnd = wireLowerEnd ^ stage;
                    if (wireUpperEnd > wireLowerEnd){
                        if (((wireLowerEnd & runSize) == 0 && arr[wireLowerEnd] > arr[wireUpperEnd]) // sort ascending
                            || ((wireLowerEnd & runSize) != 0 && arr[wireLowerEnd] < arr[wireUpperEnd])) // sort descending
                            std::swap(arr[wireLowerEnd], arr[wireUpperEnd]);
                    }
                }
            }
        }
    }

}

#endif