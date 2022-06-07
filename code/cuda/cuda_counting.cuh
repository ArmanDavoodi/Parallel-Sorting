#ifndef CUDA_SORT_COUNTING
#define CUDA_SORT_COUNTING

#include "cuda_sorts_headers.cuh"

namespace cuda_par {
    constexpr int cumulativeSumThreadsPerBlock = 1024;
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

    // k should be either a power of 2 or a power of 2 + 1
    __global__ void cumulativeSumKernel(int* d_count, int k, int N) {
        int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (thread_index < k / 2) {

            for (int stepSize = 1; stepSize <= k >> 1; stepSize <<= 1) {
                int i = (thread_index + 1) * stepSize * 2 - 1;
                if (i < k)
                    d_count[i] += d_count[i - stepSize];
                
                __syncthreads();
            }
            
            for (int stepSize = k >> 2; stepSize > 0; stepSize >>= 1) {
                int i = (thread_index + 1) * stepSize * 2 - 1;
                if (i + stepSize < k)
                    d_count[i + stepSize] += d_count[i];
                
                __syncthreads();
            }
        }
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

    cudaError_t countingSort(int* host_arr, int N, int min, int max, double& deltaTime) { // min and max should be in a way that make k a power of 2 or a power of 2 + 1
        cudaError_t cudaStatus;
        cudaEvent_t start, stop;
        int* device_arr = NULL;
        int k = max - min + 1;
        int* d_count = NULL; 
        int* h_count = new int[k];

        printf("\nN = %d, k = %d\n", N, k);

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

        cudaDeviceSynchronize();
        cudaMemcpy(h_count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
        printf("count at first:\n");
        for (int i = 0; i < k; ++i)
            printf("%d ", h_count[i]);
        printf("\n");
        
        cudaStatus = cudaInitialization(device_arr, host_arr, start, stop, N);

        countKernel<<<N / countKerThreadsPerBlock + (N % countKerThreadsPerBlock > 0)
            , std::min(countKerThreadsPerBlock, N)>>>(device_arr, d_count, N, min);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("countKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaDeviceSynchronize();
        cudaMemcpy(h_count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
        printf("count after counting:\n");
        for (int i = 0; i < k; ++i)
            printf("%d ", h_count[i]);
        printf("\n");

        cumulativeSumKernel<<<k / cumulativeSumThreadsPerBlock + (k % cumulativeSumThreadsPerBlock > 0)
            , std::min(cumulativeSumThreadsPerBlock, k)>>>(d_count, k, N);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("cumulativeSumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }

        cudaDeviceSynchronize();
        cudaMemcpy(h_count, d_count, k * sizeof(int), cudaMemcpyDeviceToHost);
        printf("count after cumulative sum:\n");
        for (int i = 0; i < k; ++i)
            printf("%d ", h_count[i]);
        printf("\n");

        printf("lower_bounds:\n");
        for (int i = 0; i < N; ++i)
            printf("%d ", lower_bound(h_count, k, i));
        printf("\n");


        printf("\n");

        countSortKernel<<<N / countSortThreadsPerBlock + (N % countSortThreadsPerBlock > 0)
            , std::min(countSortThreadsPerBlock, N)>>>(device_arr, d_count, N, k, min);

        cudaEnding(host_arr, device_arr, start, stop, deltaTime, N);

        return cudaStatus;
    }

}

#endif