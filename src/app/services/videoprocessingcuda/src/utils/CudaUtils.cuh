#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

template<typename T>
__global__ void fillArrayKernel(T* array, T value, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        array[index] = value;
    }
}

#endif //!CUDA_UTILS_H