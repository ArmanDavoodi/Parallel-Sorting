#ifndef CUDA_SORT_ODD_EVEN
#define CUDA_SORT_ODD_EVEN

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int oddEvenThreadsPerBlock = 1024;

    template<typename Num>
    __global__ void cudaOddEvenKernel(Num* device_arr, int startIdx, int N, int nt) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (index < nt) {
            index = 2 * index + startIdx;
            if (index < N - 1 && device_arr[index] > device_arr[index + 1])
                swap(device_arr[index], device_arr[index + 1]);
        }
    }

    template<typename Num>
    cudaError_t oddEvenSort(Num* host_arr, int N, double& deltaTime) {
        cudaError_t cudaStatus;
        Num* device_arr = NULL;
        cudaEvent_t start, stop;
        int numberOfNeededThreads = (N - 1) / 2 + (N - 1) % 2;

        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        for (int phase = 0; phase < N; ++phase) {
            cudaOddEvenKernel<<<numberOfNeededThreads / oddEvenThreadsPerBlock + (numberOfNeededThreads % oddEvenThreadsPerBlock > 0)
                , std::min(numberOfNeededThreads, oddEvenThreadsPerBlock)>>>(device_arr, phase % 2, N, numberOfNeededThreads);
        }
        
        cudaStatus = cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        return cudaStatus;
    }

}

#endif