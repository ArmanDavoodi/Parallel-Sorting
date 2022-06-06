#ifndef CUDA_SORT_BITONIC
#define CUDA_SORT_BITONIC

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int bitonicThreadsPerBlock = 1024;

    template<typename Num>
    __global__ void cudaBitonicKernel(Num* device_arr, int runSize, int stage, int N) {
        int wireLowerEnd = blockIdx.x * blockDim.x + threadIdx.x;
        int wireUpperEnd = wireLowerEnd ^ stage;

        if (wireUpperEnd > wireLowerEnd)
            if (((wireLowerEnd & runSize) == 0 && device_arr[wireLowerEnd] > device_arr[wireUpperEnd]) // sort ascending
                    || ((wireLowerEnd & runSize) != 0 && device_arr[wireLowerEnd] < device_arr[wireUpperEnd])) // sort descending
                    swap(device_arr[wireLowerEnd], device_arr[wireUpperEnd]);
    }

    // only works on arrays of size 2^k
    template<typename Num>
    cudaError_t bitonicSort(Num* host_arr, int N, double& deltaTime) {        
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        Num* device_arr = NULL;

        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        // size of the runs being sorted
        for (int runSize = 2; runSize <= N; runSize = 2*runSize) {
            for (int stage = runSize / 2; stage > 0; stage /= 2) {
                cudaBitonicKernel<<<N / bitonicThreadsPerBlock + (N % bitonicThreadsPerBlock > 0)
                    , std::min(N, bitonicThreadsPerBlock)>>>(device_arr, runSize, stage, N);
            }
        } 
        
        cudaStatus = cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        return cudaStatus;
    }

}

#endif