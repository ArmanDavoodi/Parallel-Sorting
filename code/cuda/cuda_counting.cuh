#ifndef CUDA_SORT_COUNTING
#define CUDA_SORT_COUNTING

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int prefixSumThreadsPerBlock = 128;
    constexpr int countKerThreadsPerBlock = 1024;
    constexpr int putValThreadsPerBlock = 1024;
    constexpr int countSortThreadsPerBlock = 1024;

    __host__ __device__ int lower_bound(int* arr, int N, int value) {
        int mid, low = 0, high = N;

        while(low < high) {
            mid = low + (high - low) / 2;

            if (value <= arr[mid])
                high = mid;
            else
                low = mid + 1;
        }

        if (low < N && arr[low] < value)
            ++low;
        
        return low;
    }  

    __global__ void blockPrefixSumKernel(int* d_arr, int* sums, int N) {
        int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ int temp_arr[prefixSumThreadsPerBlock];

        temp_arr[threadIdx.x] = thread_index < N ? d_arr[thread_index] : 0;

        __syncthreads();

        for (int stepSize = 1; stepSize <= prefixSumThreadsPerBlock >> 1; stepSize <<= 1) {
            int i = (threadIdx.x + 1) * stepSize * 2 - 1;
            if (i < prefixSumThreadsPerBlock)
                temp_arr[i] += temp_arr[i - stepSize];
            
            __syncthreads();
        }
        
        for (int stepSize = prefixSumThreadsPerBlock >> 2; stepSize > 0; stepSize >>= 1) {
            int i = (threadIdx.x + 1) * stepSize * 2 - 1;
            if (i + stepSize < prefixSumThreadsPerBlock)
                temp_arr[i + stepSize] += temp_arr[i];
            
            __syncthreads();
        }

        if (threadIdx.x == 0)
            sums[blockIdx.x] = temp_arr[prefixSumThreadsPerBlock - 1];
        
        d_arr[thread_index] = (thread_index < N) * temp_arr[threadIdx.x];
    }

    __global__ void addPrefixSumKernel(int* d_arr, int* sums, int N) {
        int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_index >= prefixSumThreadsPerBlock && thread_index < N)
            d_arr[thread_index] += sums[blockIdx.x - 1];
    }

    __global__ void countKernel(int* device_arr, int* count, int N, int min) {
        int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_index < N) {
            atomicAdd(&count[device_arr[thread_index] - min], 1);
        }
    }

    __global__ void putValKernel(int* device_memory, int N, int value) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < N)
            device_memory[index] = value;
    }

    __global__ void countSortKernel(int* device_arr, int* count, int N, int k, int min) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < N) {
            int valIdx = lower_bound(count, k, index + 1);
            device_arr[index] = valIdx + min;
        }
    }

    cudaError_t parallelPrefixSum(int* d_arr, int N) {
        int* sums = NULL;
        int numberOfBlocks = N / prefixSumThreadsPerBlock + (N % prefixSumThreadsPerBlock > 0);
        cudaError_t cudaStatus;
        
        cudaStatus = cudaMalloc((void**)&sums, numberOfBlocks * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            printf("Error: could not allocate memory for sum array(N = %d, numberOfBlocks = %d): \n%s\n"
                , N, numberOfBlocks, cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        blockPrefixSumKernel<<<numberOfBlocks, prefixSumThreadsPerBlock>>>(d_arr, sums, N);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("Error: blockPrefixSumKernel launch failed(N = %d, numberOfBlocks = %d): \n%s\n"
                , N, numberOfBlocks, cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        if (numberOfBlocks > 1) {
            cudaStatus = parallelPrefixSum(sums, numberOfBlocks);
            if (cudaStatus != cudaSuccess)
                return cudaStatus;

            addPrefixSumKernel<<<numberOfBlocks, prefixSumThreadsPerBlock>>>(d_arr, sums, N);
            if (cudaStatus != cudaSuccess) {
                printf("Error: addPrefixSumKernel launch failed(N = %d, numberOfBlocks = %d): \n%s\n"
                    , N, numberOfBlocks, cudaGetErrorString(cudaStatus));
                return cudaStatus;
            }
        }

        return cudaStatus;
    }

    cudaError_t countingSort(int* host_arr, int N, int min, int max, double& deltaTime) {
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        int* device_arr = NULL;
        int k = max - min + 1;
        int* d_count = NULL; 
        int* h_count = new int[k];

        cudaStatus = cudaMalloc((void**)&d_count, k * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            printf("\nError: device memory allocation for count array failed:\n%s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        putValKernel<<<k / putValThreadsPerBlock + (k % putValThreadsPerBlock > 0)
            , std::min(putValThreadsPerBlock, k)>>>(d_count, k, 0);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("puVal launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        
        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        countKernel<<<N / countKerThreadsPerBlock + (N % countKerThreadsPerBlock > 0)
            , std::min(countKerThreadsPerBlock, N)>>>(device_arr, d_count, N, min);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("countKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        parallelPrefixSum(d_count, k);

        countSortKernel<<<N / countSortThreadsPerBlock + (N % countSortThreadsPerBlock > 0)
            , std::min(countSortThreadsPerBlock, N)>>>(device_arr, d_count, N, k, min);

        cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        return cudaStatus;
    }

}

#endif