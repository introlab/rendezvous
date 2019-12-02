#include "cuda_fisheye_dewarper.h"

#include "model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh"

namespace Model
{
namespace
{
const int BLOCK_SIZE = 1024;

__device__ void dewarpImagePixel(const Image& src, const Image& dst, int srcIndex, int dstIndex)
{
    if (srcIndex < int(src.size) &&
        srcIndex > 0)    // Don't need to check the other ones, as they will be ok if these are
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

__device__ void dewarpImagePixelFiltered(const Image& src, const Image& dst, const LinearPixelFilter& linearPixelFilter,
                                         int dstIndex)
{
    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        for (int channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + channelIndex;
            dst.deviceData[dstChannelIndex] =
                src.deviceData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio;
            dst.deviceData[dstChannelIndex] +=
                src.deviceData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio;
            dst.deviceData[dstChannelIndex] +=
                src.deviceData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio;
            dst.deviceData[dstChannelIndex] +=
                src.deviceData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio;
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

        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, dst);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        int srcIndex = getSourcePixelIndexDevice(srcPosition, src) * 3;  // For now only can dewarp in RGB format
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
        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, dst);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilterDevice(srcPosition, src);
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

}    // namespace

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

void CudaFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst,
                                              const DewarpingParameters& params) const
{
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params);
}

void CudaFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst,
                                              const FilteredDewarpingMapping& mapping) const
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
}    // namespace Model