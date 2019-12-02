#include "cuda_darknet_fisheye_dewarper.h"

#include "model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh"
#include "model/stream/utils/cuda_utils.cuh"

namespace Model
{
namespace
{
const int BLOCK_SIZE = 1024;

__device__ void dewarpImagePixelNormalized(const Image& src, const ImageFloat& dst, int srcIndex, int dstIndex)
{
    int size = dst.width * dst.height;

    if (srcIndex < int(src.size) &&
        srcIndex > 0)    // Don't need to check the other ones, as they will be ok if these are
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

__device__ void dewarpImagePixelFilteredNormalized(const Image& src, const ImageFloat& dst,
                                                   const LinearPixelFilter& linearPixelFilter, int dstIndex)
{
    int size = dst.width * dst.height;

    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        for (int channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + size * channelIndex;
            dst.deviceData[dstChannelIndex] =
                (src.deviceData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] +=
                (src.deviceData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] +=
                (src.deviceData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio) / 255.f;
            dst.deviceData[dstChannelIndex] +=
                (src.deviceData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio) / 255.f;
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
        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, dst);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        int srcIndex = getSourcePixelIndexDevice(srcPosition, src) * 3;  // For now only can dewarp in RGB format
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
        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, dst);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilterDevice(srcPosition, src);
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

}    // namespace

CudaDarknetFisheyeDewarper::CudaDarknetFisheyeDewarper(cudaStream_t stream, float outputAspectRatio)
    : mappingFiller_(stream)
    , outputAspectRatio_(outputAspectRatio)
    , stream_(stream)
{
}

void CudaDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst,
                                             const DewarpingParameters& params) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst,
                                             const DewarpingMapping& mapping) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst,
                                                     const DewarpingParameters& params) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(dst, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, params, offset);
}

void CudaDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst,
                                                     const FilteredDewarpingMapping& mapping) const
{
    int offset = calculateOffset(dst, outputAspectRatio_);
    int blockCount = calculateKernelBlockCount(mapping, BLOCK_SIZE);
    dewarpImageFilteredKernel<<<blockCount, BLOCK_SIZE, 0, stream_>>>(src, dst, mapping, offset);
}

void CudaDarknetFisheyeDewarper::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                                      const DewarpingMapping& mapping) const
{
    mappingFiller_.fillDewarpingMapping(src, params, mapping);
}

void CudaDarknetFisheyeDewarper::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
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
}    // namespace Model