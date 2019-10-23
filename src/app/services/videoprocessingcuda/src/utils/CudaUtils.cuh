#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>
#include <stdexcept>

#include <cuda_runtime.h>

template<typename T>
__global__ void fillArrayKernel(T* array, T value, std::size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        array[index] = value;
    }
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        std::string error = "Error : ";
        throw std::runtime_error(error.append(cudaGetErrorString(result)));
    }
    return result;
}

#endif //!CUDA_UTILS_H