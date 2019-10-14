#ifndef ERROR_HELPER_H
#define ERROR_HELPER_H

#ifndef NO_CUDA

#include <string>
#include <stdexcept>

#include "cuda_runtime.h"

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        std::string error = "Error : ";
        throw std::runtime_error(error.append(cudaGetErrorString(result)));
    }
    return result;
}
#endif

#endif // !ERROR_HELPER_H
