#ifndef CUDA_DEWARPING_HELPER_H
#define CUDA_DEWARPING_HELPER_H

#include <cuda_runtime.h>

#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/dim3.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"

namespace Model
{

// __device__ Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index);
// __device__ LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim3<int>& dim);
// __device__ int calculateSourcePixelIndex(const Point<float>& pixel, const Dim3<int>& dim);
// int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize);

} // Model

#endif // !CUDA_DEWARPING_HELPER_H
