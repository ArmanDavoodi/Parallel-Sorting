#ifndef OMP_SORT_BITONIC
#define OMP_SORT_BITONIC

#include "omp_sorts_headers.h"

namespace omp_par {
    // only works on arrays of size 2^k
    template<typename Num>
    void bitonicSort(Num* arr, int N, double& deltaTime) {        
        double t = omp_get_wtime();
        
        // size of the runs being sorted
        #pragma omp parallel num_threads(P) // use or not TODO ###############################
        for (int runSize = 2; runSize <= N; runSize = 2*runSize) {
            for (int stage = runSize / 2; stage > 0; stage /= 2) {
                #pragma omp for // TODO scheduling ########################
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
        
        deltaTime += omp_get_wtime() - t;
    }

}

#endif