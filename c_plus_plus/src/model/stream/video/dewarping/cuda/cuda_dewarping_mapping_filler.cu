#include "cuda_dewarping_mapping_filler.h"

//#include "model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh"

namespace
{
const int BLOCK_SIZE = 1024;

// THIS SHOULD BE IN CudaDewarpingHelper, BUT SWIG IS A BITCH
#include "model/stream/utils/math/math_constants.h"
#include "model/stream/utils/models/dim3.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"

__device__ Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index)
{
    float x = index % dst.width;
    float y = index / dst.width;

    float textureCoordsX = x / dst.width;
    float textureCoordsY = y / dst.height;

    float heightFactor = (1 - ((params.bottomOffset + params.topOffset) / params.dewarpHeight)) * textureCoordsY +
                         params.topOffset / params.dewarpHeight;
    float factor = params.outRadiusDiff * (1 - params.bottomDistorsionFactor) * __sinf(math::PI * textureCoordsX) *
                   __sinf((math::PI * heightFactor) / 2.0);
    float radius = textureCoordsY * params.dewarpHeight + params.inRadius + factor +
                   (1 - textureCoordsY) * params.topOffset - textureCoordsY * params.bottomOffset;
    float theta = ((textureCoordsX * params.dewarpWidth) + params.xOffset) / params.centerRadius;

    Point<float> srcPixelPosition;
    srcPixelPosition.x = params.xCenter + radius * __sinf(theta);
    srcPixelPosition.y = params.yCenter + radius * __cosf(theta);

    return srcPixelPosition;
}

__device__ LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim2<int>& dim)
{
    int xRoundDown = int(pixel.x);
    int yRoundDown = int(pixel.y);
    float xRatio = pixel.x - xRoundDown;
    float yRatio = pixel.y - yRoundDown;
    float xOpposite = 1 - xRatio;
    float yOpposite = 1 - yRatio;

    LinearPixelFilter linearPixelFilter;

    linearPixelFilter.pc1.index = (xRoundDown + (yRoundDown * dim.width)) * 3;
    linearPixelFilter.pc2.index = linearPixelFilter.pc1.index + 3;
    linearPixelFilter.pc3.index = linearPixelFilter.pc1.index + dim.width * 3;
    linearPixelFilter.pc4.index = linearPixelFilter.pc2.index + dim.width * 3;

    linearPixelFilter.pc1.ratio = xOpposite * yOpposite;
    linearPixelFilter.pc2.ratio = xRatio * yOpposite;
    linearPixelFilter.pc3.ratio = xOpposite * yRatio;
    linearPixelFilter.pc4.ratio = xRatio * yRatio;

    return linearPixelFilter;
}

__device__ int calculateSourcePixelIndex(const Point<float>& pixel, const Dim2<int>& dim)
{
    return (int(pixel.x) + int(pixel.y) * dim.width) * 3;
}

int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize)
{
    return (dim.width * dim.height + blockSize - 1) / blockSize;
}

// END OF RANT

__global__ void fillDewarpingMappingKernel(Dim2<int> src, DewarpingParameters params, DewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < mapping.width * mapping.height)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(mapping, params, index);
        mapping.deviceData[index] = calculateSourcePixelIndex(srcPosition, src);
    }
}

__global__ void fillFilteredDewarpingMappingKernel(Dim2<int> src, DewarpingParameters params,
                                                   FilteredDewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < mapping.width * mapping.height)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(mapping, params, index);
        mapping.deviceData[index] = calculateLinearPixelFilter(srcPosition, src);
    }
}

}    // namespace

CudaDewarpingMappingFiller::CudaDewarpingMappingFiller(cudaStream_t stream)
    : stream_(stream)
{
}

void CudaDewarpingMappingFiller::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                                      const DewarpingMapping& mapping) const
{
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    fillDewarpingMappingKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, params, mapping);
}

void CudaDewarpingMappingFiller::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                                              const FilteredDewarpingMapping& mapping) const
{
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    fillFilteredDewarpingMappingKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, params, mapping);
}
