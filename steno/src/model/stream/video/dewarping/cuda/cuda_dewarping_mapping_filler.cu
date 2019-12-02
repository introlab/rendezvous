#include "cuda_dewarping_mapping_filler.h"

#include "model/stream/video/dewarping/cuda/cuda_dewarping_helper.cuh"

namespace Model
{
namespace
{
const int BLOCK_SIZE = 1024;

__global__ void fillDewarpingMappingKernel(Dim2<int> src, DewarpingParameters params, DewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < mapping.width * mapping.height)
    {
        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, mapping);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        mapping.deviceData[index] = getSourcePixelIndexDevice(srcPosition, src) * 3;  // For now only can dewarp in RGB format
    }
}

__global__ void fillFilteredDewarpingMappingKernel(Dim2<int> src, DewarpingParameters params,
                                                   FilteredDewarpingMapping mapping)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < mapping.width * mapping.height)
    {
        Point<float> normalizedPixel = getNormalizedPixelFromIndexDevice(index, mapping);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixelDevice(normalizedPixel, params);
        mapping.deviceData[index] = calculateLinearPixelFilterDevice(srcPosition, src);
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
}    // namespace Model