#ifndef OMP_SORT_MERGE
#define OMP_SORT_MERGE

#include "omp_sorts_headers.h"
#include "../sequential/insertion.h"
#include "../sequential/merge.h"

namespace omp_par {
    // bottom-up implementation
    template<typename Num>
    void mergeSort(Num* arr, int N, double& deltaTime) {    
        double t = omp_get_wtime();

        register int currentSize = seq::getMinRunSize(N), i;

        // ad-hoc -> sort smaller blocks using insertion sort
        if (currentSize > 1) {
            #pragma omp parallel for num_threads(P) schedule(dynamic) // use dynamic scheduling for load balancing
            for (i = 0; i < N; i += currentSize)
                seq::insertionSort(arr + i, i + currentSize < N ? currentSize : N - i);
        }
        
        #pragma omp parallel num_threads(P) // use or not TODO ##################################
        for (; currentSize < N; currentSize *= 2) {
            #pragma omp for schedule(dynamic)
            for (i = 0; i < N - 1; i += 2*currentSize) {
                int mid = std::min(i + currentSize, N-1);
                int right = std::min(i + 2*currentSize - 1, N-1);

                if (arr[mid - 1] > arr[mid]) // merge two blocks if they are not already sorted
                    seq::merge(arr, i, mid, right);
            }
        }

        deltaTime += omp_get_wtime() - t;
    }

}

#endif