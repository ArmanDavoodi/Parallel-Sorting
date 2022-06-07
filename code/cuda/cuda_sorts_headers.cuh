#ifndef CUDA_SORT_HEAD
#define CUDA_SORT_HEAD

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>

namespace cuda_par {
    template<typename T>
    __device__ void swap(T& a, T& b) {
        T c = a;
        a = b;
        b = c;
    }

    template<typename Num>
    cudaError_t cudaInitialization(Num*& device_arr, Num* host_arr, cudaEvent_t& start, cudaEvent_t& stop, int N) {
        cudaError_t cudaStatus;

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

        return cudaStatus;
    }

    template<typename Num>
    cudaError_t cudaEnding(Num*& host_arr, Num* device_arr, cudaEvent_t& start, cudaEvent_t& stop, double& deltaTime, int N) {
        cudaError_t cudaStatus;

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

        cudaStatus = cudaEventDestroy(start);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: start event could not be destroyed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaStatus = cudaEventDestroy(stop);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: stop event could not be destroyed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

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