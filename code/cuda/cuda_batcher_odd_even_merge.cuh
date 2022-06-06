#ifndef CUDA_SORT_BOE_MERGE
#define CUDA_SORT_BOE_MERGE

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int oddEvenMergeThreadsPerBlock = 1024;

    template<typename Num>
    __global__ void cudaOddEvenMergeKernel(Num* device_arr, int N, int p, int k) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        index = (index / k) * 2 * k + k % p + index % k;

        if (index / (2*p) == (index + k) / (2*p) && device_arr[index] > device_arr[index + k])
            swap(device_arr[index], device_arr[index + k]);
    }

    // only works on arrays of size 2^k
    template<typename Num>
    cudaError_t batcherOddEvenMergeSort(Num* host_arr, int N, double& deltaTime) {
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        Num* device_arr = NULL;

        int srn = ceil(sqrt(N));
        

        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        for (int p = 1; p < N; p = 2*p) {
            for (int k = p; k > 0; k /= 2) {
                int numberOfThreads = k * ((((N - k) - (k % p)) / (2 * k)) + (((N - k) - (k % p)) % (2 * k) > 0));

                // printf("\n%d %d _ %d _ %d %d\n", numberOfThreads / oddEvenMergeThreadsPerBlock 
                //     + (numberOfThreads % oddEvenMergeThreadsPerBlock > 0)
                //     , std::min(oddEvenMergeThreadsPerBlock, numberOfThreads), numberOfThreads, p, k);
                if (numberOfThreads > 0)
                    cudaOddEvenMergeKernel<<<numberOfThreads / oddEvenMergeThreadsPerBlock 
                        + (numberOfThreads % oddEvenMergeThreadsPerBlock > 0)
                        , std::min(oddEvenMergeThreadsPerBlock, numberOfThreads)>>>(device_arr, N, p, k);
                // #pragma omp for // TODO where to use + scheduling ##############################
                // int s = 0, d = 0;
                // for (int j = k % p; j < N - k; j += 2*k) {
                //     ++d;
                //     for (int i = 0; i < k; ++i) {
                //         ++s;
                //         // if ((i + j) / (2 * p) == (i + j + k) / (2 * p))
                //         //     if (arr[i + j] > arr[i + j + k])
                //         //         std::swap(arr[i + j], arr[i + j + k]);
                //     }
                // }

                // printf("%d %d %d\n", s, d, ((N - k) - (k % p)) / (2*k) + (((N - k) - (k % p)) % (2*k) > 0));
            }
        }

        cudaStatus = cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        return cudaStatus;
    }

}

#endif