#ifndef OMP_SORT_COUNTING
#define OMP_SORT_COUNTING

#include "omp_sorts_headers.h"

namespace omp_par {
    void countingSort(int* arr, int N, int min, int max, double& deltaTime) {
        double t = omp_get_wtime();

        int k = max - min + 1;
        int* count = new int[k];
        omp_lock_t* countLock = new omp_lock_t[k];
        int* input = new int[N];
        
        #pragma omp parallel for num_threads(P)
        for (int i = 0; i < k; ++i) {
            omp_init_lock(countLock + i);
            count[i] = 0;
        }

        #pragma omp parallel for num_threads(P)
        for (int i = 0; i < N; ++i)
            input[i] = arr[i];

        #pragma omp parallel for num_threads(P)
        for (int i = 0; i < N; ++i) {
            #pragma omp atomic
            ++count[input[i] - min];
        }

        int pTemp[P] = {0};
        #pragma omp parallel for num_threads(P) // compute sum function for each block
        for (int i = 1; i < k; ++i) {
            int rank = omp_get_thread_num();
            if (pTemp[rank] != -1)
                pTemp[rank] = -1;
            else
                count[i] += count[i - 1];
        }

        int remaining = k % P - 1;
        pTemp[0] = count[0]; // sum of previous blocks' elements
        for (int i = k/P, r = 1; i < k; i += k/P, ++r) {
            if (remaining > 0) {
                ++i;
                --remaining;
            }
            pTemp[r] = count[i] + pTemp[r - 1];
        }

        // compute the sum array
        #pragma omp parallel for num_threads(P)
        for (int i = 1; i < k; ++i) {
            int rank = omp_get_thread_num();
                count[i] += pTemp[rank];
        }

        // sort
        #pragma omp parallel for num_threads(P)
        for (int i = N - 1; i >= 0; --i) {
            int index = input[i] - min;

            omp_set_lock(countLock + index);
            arr[count[index] - 1] = input[i];
            --count[index];
            omp_unset_lock(countLock + index);
        }

        #pragma omp parallel for num_threads(P)
        for (int i = 0; i < k; ++i) {
            omp_destroy_lock(countLock + i);
        }

        delete[] input;
        delete[] count;
        delete[] countLock;

        deltaTime += omp_get_wtime() - t;
    }

}

#endif