#ifndef CUDA_SORT_MERGE
#define CUDA_SORT_MERGE

#include "cuda_sorts_headers.cuh"
#include "../sequential/merge.h"

namespace cuda_par {
    constexpr int insertionSortThreadsPerBlock = 1024; // tune
    constexpr int mergeThreadsPerBlock = 1024;

    template<typename Num>
    __global__ void insertionSortKernel(Num* device_arr, int currentSize, int N, int nt) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < nt) {
            index *= currentSize;
            int size = index + currentSize < N ? currentSize : N - index;
            device_arr += index;

            for (int i = 1; i < size; ++i) {
                int j;
                Num temp = device_arr[i];
                for (j = i; (j > 0) && (temp < device_arr[j-1]); --j) {
                    device_arr[j] = device_arr[j-1];
                }
                device_arr[j] = temp;
            }
        }
    }

    template<typename Num>
    __global__ void mergeKernel(Num* device_arr, Num* temp_arr, int currentSize, int N, int nt) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index < nt) {
            index *= 2*currentSize;
            int mid = N - 1 > index + currentSize ? index + currentSize : N - 1;
            int right = N - 1 > index + 2*currentSize - 1 ? index + 2*currentSize - 1 : N - 1;

            if (device_arr[mid - 1] > device_arr[mid]) {
                
                int nl = mid - index, nr = right - mid + 1;
                Num* left_arr = temp_arr + index;
                Num* right_arr = temp_arr + mid;
                int i, j, k;

                // copy the two blocks into two temp arrays
                for (i = 0; i < nl; ++i)
                    left_arr[i] = device_arr[i + index];
                for (i = 0; i < nr; ++i)
                    right_arr[i] = device_arr[i + mid];
                
                // merge algorithm
                i = 0;
                j = 0;
                k = index;
                while (i < nl && j < nr) {
                    if (left_arr[i] <= right_arr[j]) {
                        device_arr[k] = left_arr[i];
                        ++i;
                    }
                    else {
                        device_arr[k] = right_arr[j];
                        ++j;
                    }
                    ++k;
                }

                // put the remaining elements in the array
                while (i < nl) {
                    device_arr[k] = left_arr[i];
                    ++i;
                    ++k;
                }

                while (j < nr) {
                    device_arr[k] = right_arr[j];
                    ++j;
                    ++k;
                }
            }
        }
    }

    // bottom-up implementation
    template<typename Num>
    cudaError_t mergeSort(Num* host_arr, int N, double& deltaTime) {    
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        Num* device_arr = NULL;
        Num* temp_arr = NULL; 

        cudaStatus = cudaMalloc((void**)&temp_arr, N * sizeof(Num));
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device memory allocation for temp array failed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }       

        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        register int currentSize = seq::getMinRunSize(N);

        // ad-hoc -> sort smaller blocks using insertion sort
        if (currentSize > 1) {
            int nt = N / currentSize + (N % currentSize > 0);
            insertionSortKernel<<<nt / insertionSortThreadsPerBlock + (nt % insertionSortThreadsPerBlock > 0)
                , std::min(nt, insertionSortThreadsPerBlock)>>>(device_arr, currentSize, N, nt);
        }
        
        for (; currentSize < N; currentSize *= 2) {
            int nt = (N - 1) / (2 * currentSize) + ((N - 1) % (2 * currentSize) > 0);
            mergeKernel<<<nt / mergeThreadsPerBlock + (nt % mergeThreadsPerBlock > 0)
                , std::min(nt, mergeThreadsPerBlock)>>>(device_arr, temp_arr, currentSize, N, nt);
        }

        cudaStatus = cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        cudaStatus = cudaFree(temp_arr);
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device memory temp array could not be freed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        return cudaStatus;
    }

}

#endif