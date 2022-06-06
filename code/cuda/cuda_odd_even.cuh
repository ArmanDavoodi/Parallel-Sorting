#ifndef CUDA_SORT_ODD_EVEN
#define CUDA_SORT_ODD_EVEN

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int oddEvenThreadsPerBlock = 1024;

    template<typename Num>
    __global__ void cudaOddEvenKernel(Num* device_arr, int startIdx, int N) {
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + startIdx;
        if (index < N - 1 && device_arr[index] > device_arr[index + 1])
            swap(device_arr[index], device_arr[index + 1]);
    }

    template<typename Num>
    cudaError_t oddEvenSort(Num* host_arr, int N, double& deltaTime) {
        cudaError_t cudaStatus;
        Num* device_arr = NULL;
        int numberOfNeededThreads = (N - 1) / 2 + (N - 1) % 2;

        cudaStatus = cudaMalloc((void**)&device_arr, N * sizeof(Num));
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device memory allocation failed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(device_arr, host_arr, N * sizeof(Num), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: host array could not be copied into the device memory:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaEvent_t start, stop;

        cudaStatus = cudaEventCreate(&start);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: start Event could not be created:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaEventCreate(&stop);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: stop Event could not be created:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaEventRecord(start, NULL);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: start Event could not be recorded:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        for (int phase = 0; phase < N; ++phase) {
            cudaOddEvenKernel<<<numberOfNeededThreads / oddEvenThreadsPerBlock + (numberOfNeededThreads % oddEvenThreadsPerBlock > 0)
                , std::min(numberOfNeededThreads, oddEvenThreadsPerBlock)>>>(device_arr, phase % 2, N);
        }
        cudaStatus = cudaEventRecord(stop, NULL);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: stop Event could not be recorded:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("\nError: kernel failed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaEventSynchronize(stop);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: cuda event synchronize failed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        float ms;
        cudaStatus = cudaEventElapsedTime(&ms, start, stop);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: could not compute elapsed time:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        deltaTime += ms / 1000;

        cudaStatus = cudaMemcpy(host_arr, device_arr, N * sizeof(Num), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device array could not be copied into the host memory:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaFree(device_arr);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device memory could not be freed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        return cudaStatus;
    }

}

#endif