#ifndef CUDA_DEWARPING_HELPER_H
#define CUDA_DEWARPING_HELPER_H

#include <cuda_runtime.h>

#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/dim2.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"

namespace Model
{

__device__ Point<float> getSourcePixelFromDewarpedImageNormalizedPixelDevice(const Point<float>& normalizedPixel, 
                                                                             const DewarpingParameters& dewarpingParameters);
__device__ LinearPixelFilter calculateLinearPixelFilterDevice(const Point<float>& pixel, const Dim2<int>& dim);

__device__ Point<float> getNormalizedPixelFromIndexDevice(int index, const Dim2<int>& dim);
__device__ int getSourcePixelIndexDevice(const Point<float>& pixel, const Dim2<int>& dim);

int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize);

}    // namespace Model

#endif // !CUDA_DEWARPING_HELPER_H
