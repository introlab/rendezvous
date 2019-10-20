#include "CudaDarknetFisheyeDewarper.h"

//#include "dewarping/cuda/CudaDewarpingHelper.cuh"
#include "utils/CudaUtils.cuh"

namespace
{

const int BLOCK_SIZE = 1024;

// THIS SHOULD BE IN CudaDewarpingHelper, BUT SWIG IS A BITCH
#include "utils/models/Point.h"
#include "utils/models/Dim3.h"
#include "dewarping/models/LinearPixelFilter.h"
#include "dewarping/models/DewarpingParameters.h"
#include "utils/math/MathConstants.h"

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

__device__ LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim3<int>& dim)
{
    int xRoundDown = int(pixel.x);
    int yRoundDown = int(pixel.y);
    float xRatio = pixel.x - xRoundDown;
    float yRatio = pixel.y - yRoundDown;
    float xOpposite = 1 - xRatio;
    float yOpposite = 1 - yRatio;

    LinearPixelFilter linearPixelFilter;
    
    linearPixelFilter.pc1.index = (xRoundDown + (yRoundDown * dim.width)) * dim.channels;
    linearPixelFilter.pc2.index = linearPixelFilter.pc1.index + dim.channels;
    linearPixelFilter.pc3.index = linearPixelFilter.pc1.index + dim.width * dim.channels;
    linearPixelFilter.pc4.index = linearPixelFilter.pc2.index + dim.width * dim.channels;

    linearPixelFilter.pc1.ratio = xOpposite * yOpposite;
    linearPixelFilter.pc2.ratio = xRatio * yOpposite;
    linearPixelFilter.pc3.ratio = xOpposite * yRatio;
    linearPixelFilter.pc4.ratio = xRatio * yRatio;

    return linearPixelFilter;
}

__device__ int calculateSourcePixelIndex(const Point<float>& pixel, const Dim3<int>& dim)
{
    return (int(pixel.x) + int(pixel.y) * dim.width) * dim.channels;
}

int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize)
{
   return (dim.width * dim.height + blockSize - 1) / blockSize;
}

// END OF RANT

__device__ void dewarpImagePixelNormalized(const Image& src, const ImageFloat& dst, int srcIndex, int dstIndex)
{
    int size = dst.width * dst.height;

    if (srcIndex < int(src.size) && srcIndex > 0) // Don't need to check the other ones, as they will be ok if these are
    {
        dst.deviceData[dstIndex] = (src.deviceData[srcIndex]) / 255.f;
        dst.deviceData[dstIndex + size] = (src.deviceData[srcIndex + 1]) / 255.f;
        dst.deviceData[dstIndex + 2 * size] = (src.deviceData[srcIndex + 2]) / 255.f;
    }
    else
    {
        dst.deviceData[dstIndex] = 0;
        dst.deviceData[dstIndex + size] = 0;
        dst.deviceData[dstIndex + 2 * size] = 0;
    }
}

__device__ void dewarpImagePixelFilteredNormalized(const Image& src, const ImageFloat& dst, const LinearPixelFilter& linearPixelFilter, int dstIndex)
{        
    int size = dst.width * dst.height;

    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        for (int channelIndex = 0; channelIndex < src.channels; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + size * channelIndex;
            dst.deviceData[dstChannelIndex] = (src.deviceData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] += (src.deviceData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] += (src.deviceData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] += (src.deviceData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio) / 255.f;
        }
    }
    else
    {
        dst.deviceData[dstIndex] = 0;
        dst.deviceData[dstIndex + size] = 0;
        dst.deviceData[dstIndex + 2 * size] = 0;
    }
}

__global__ void dewarpImageKernel(Image src, ImageFloat dst, DewarpingParameters params, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        int srcIndex = calculateSourcePixelIndex(srcPosition, src);
        dewarpImagePixelNormalized(src, dst, srcIndex, index + offset);
    }
}

__global__ void dewarpImageKernel(Image src, ImageFloat dst, DewarpingMapping mapping, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        dewarpImagePixelNormalized(src, dst, mapping.deviceData[index], index + offset);
    }
}

__global__ void dewarpImageFilteredKernel(Image src, ImageFloat dst, DewarpingParameters params, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilter(srcPosition, src);
        dewarpImagePixelFilteredNormalized(src, dst, linearPixelFilter, index + offset);
    }
}

__global__ void dewarpImageFilteredKernel(Image src, ImageFloat dst, FilteredDewarpingMapping mapping, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dst.width * dst.height)
    {
        dewarpImagePixelFilteredNormalized(src, dst, mapping.deviceData[index], index + offset);
    }
}

int calculateOffset(const Dim2<int>& dim, float aspectRatio)
{
    return ((dim.height - int(dim.height * aspectRatio)) / 2) * dim.width;
}

}

CudaDarknetFisheyeDewarper::CudaDarknetFisheyeDewarper(cudaStream_t stream, float outputAspectRatio)
    : mappingFiller_(stream)
    , outputAspectRatio_(outputAspectRatio)
    , stream_(stream)
{
}

void CudaDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst, const FilteredDewarpingMapping& mapping) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping, offset);
}

void CudaDarknetFisheyeDewarper::fillDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, 
                                                      const DewarpingMapping& mapping) const
{
    mappingFiller_.fillDewarpingMapping(src, params, mapping);
}

void CudaDarknetFisheyeDewarper::fillFilteredDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, 
                                                              const FilteredDewarpingMapping& mapping) const
{
    mappingFiller_.fillFilteredDewarpingMapping(src, params, mapping);
}

void CudaDarknetFisheyeDewarper::prepareOutputImage(ImageFloat& dst) const
{
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    fillArrayKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(dst.deviceData, 0.5f, dst.size);
}

Dim2<int> CudaDarknetFisheyeDewarper::getRectifiedOutputDim(const Dim2<int>& dst) const
{
    return Dim2<int>(dst.width, dst.height * outputAspectRatio_);
}