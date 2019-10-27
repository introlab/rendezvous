#include "cuda_fisheye_dewarper.h"

//#include "model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh"

namespace
{

const int BLOCK_SIZE = 1024;

// THIS SHOULD BE IN CudaDewarpingHelper, BUT SWIG IS A BITCH
#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/dim3.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/utils/math/math_constants.h"

__device__ Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index)
{
    float x = index % dst.width;
    float y = index / dst.width;

    float textureCoordsX = x / dst.width;
    float textureCoordsY = y / dst.height;

    float heightFactor = (1 - ((params.bottomOffset + params.topOffset) / params.dewarpHeight)) *
        textureCoordsY + params.topOffset / params.dewarpHeight;
    float factor = params.outRadiusDiff * (1 - params.bottomDistorsionFactor) *
        __sinf(math::PI * textureCoordsX) * __sinf((math::PI * heightFactor) / 2.0);
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

__device__ void dewarpImagePixel(const Image& src, const Image& dst, int srcIndex, int dstIndex)
{
    if (srcIndex < int(src.size) && srcIndex > 0) // Don't need to check the other ones, as they will be ok if these are
    {
        dst.deviceData[dstIndex] = src.deviceData[srcIndex];
        dst.deviceData[dstIndex + 1] = src.deviceData[srcIndex + 1];
        dst.deviceData[dstIndex + 2] = src.deviceData[srcIndex + 2];
    }
    else
    {
        dst.deviceData[dstIndex] = 0;
        dst.deviceData[dstIndex + 1] = 0;
        dst.deviceData[dstIndex + 2] = 0;
    }
}

__device__ void dewarpImagePixelFiltered(const Image& src, const Image& dst, const LinearPixelFilter& linearPixelFilter, int dstIndex)
{
    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        for (int channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + channelIndex;
            dst.deviceData[dstChannelIndex] = src.deviceData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio;
            dst.deviceData[dstChannelIndex] += src.deviceData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio;
            dst.deviceData[dstChannelIndex] += src.deviceData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio;
            dst.deviceData[dstChannelIndex] += src.deviceData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio;
        }
    }
    else
    {
        dst.deviceData[dstIndex] = 0;
        dst.deviceData[dstIndex + 1] = 0;
        dst.deviceData[dstIndex + 2] = 0;
    }
}

__global__ void dewarpImageKernel(Image src, Image dst, DewarpingParameters params)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        int dstIndex = index * 3;
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        int srcIndex = calculateSourcePixelIndex(srcPosition, src);
        dewarpImagePixel(src, dst, srcIndex, dstIndex);
    }
}

__global__ void dewarpImageKernel(Image src, Image dst, DewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        int dstIndex = index * 3;
        dewarpImagePixel(src, dst, mapping.deviceData[index], dstIndex);
    }
}

__global__ void dewarpImageFilteredKernel(Image src, Image dst, DewarpingParameters params)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        int dstIndex = index * 3;
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilter(srcPosition, src);
        dewarpImagePixelFiltered(src, dst, linearPixelFilter, dstIndex);
    }
}

__global__ void dewarpImageFilteredKernel(Image src, Image dst, FilteredDewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        int dstIndex = index * 3;
        dewarpImagePixelFiltered(src, dst, mapping.deviceData[index], dstIndex);
    }
}

}

CudaFisheyeDewarper::CudaFisheyeDewarper(const cudaStream_t& stream)
    : mappingFiller_(stream)
    , stream_(stream)
{
}

void CudaFisheyeDewarper::dewarpImage(const Image& src, const Image& dst, const DewarpingParameters& params) const
{
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params);
}

void CudaFisheyeDewarper::dewarpImage(const Image& src, const Image& dst, const DewarpingMapping& mapping) const
{
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping);
}

void CudaFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst, const DewarpingParameters& params) const
{
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params);
}

void CudaFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst, const FilteredDewarpingMapping& mapping) const
{
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping);
}

void CudaFisheyeDewarper::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                               const DewarpingMapping& mapping) const
{
    mappingFiller_.fillDewarpingMapping(src, params, mapping);
}

void CudaFisheyeDewarper::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                       const FilteredDewarpingMapping& mapping) const
{
    mappingFiller_.fillFilteredDewarpingMapping(src, params, mapping);
}

