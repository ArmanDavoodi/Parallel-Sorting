#ifndef SORT_COUNTING
#define SORT_COUNTING

#include "sorts_headers.h"

namespace seq {
    void countingSort(int* arr, int N, int min, int max, double& deltaTime) {
        double t = omp_get_wtime(); 

        int k = max - min + 1;
        register int i;
        int* count = new int[k];
        int* input = new int[N];
        
        for (i = 0; i < N; ++i)
            input[i] = arr[i];
        
        for (int i = 0; i < k; ++i)
            count[i] = 0;

        for (i = 0; i < N; ++i)
            ++count[input[i] - min];
        
        for (i = 1; i < k; ++i)
            count[i] += count[i - 1];

        for (i = N - 1; i >= 0; --i) {
            arr[count[input[i] - min] - 1] = input[i];
            --count[input[i] - min];
        }

        delete[] input;
        delete[] count;

        deltaTime += omp_get_wtime() - t;
    }

}

#endif