#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda_runtime.h>

namespace math
{

template<typename T>
__device__ T clamp(T value, const T& min, const T& max)
{
    if (value < min) value = min;
    if (value > max) value = max;
    return value;
}

}

#endif //!CUDA_HELPERS_H