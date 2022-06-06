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
}

#endif