#ifndef CUDA_DEWARPING_HELPER_H
#define CUDA_DEWARPING_HELPER_H

#include <cuda_runtime.h>

#include "utils/models/Point.h"
#include "utils/models/Dim3.h"
#include "dewarping/models/LinearPixelFilter.h"
#include "dewarping/models/DewarpingParameters.h"

__device__ Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index);
__device__ LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim3<int>& dim);
__device__ int calculateSourcePixelIndex(const Point<float>& pixel, const Dim3<int>& dim);
int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize);

#endif // !CUDA_DEWARPING_HELPER_H