#include "cpu_darknet_fisheye_dewarper.h"

#include "model/stream/video/dewarping/dewarping_helper.h"
#include "model/stream/utils/array_utils.h"

namespace Model
{

namespace
{

void dewarpImagePixelNormalized(const Image& src, const ImageFloat& dst, int srcIndex, int dstIndex)
{
    int size = dst.width * dst.height;
    
    if (srcIndex < int(src.size) && srcIndex > 0) // Don't need to check the other ones, as they will be ok if these are
    {
        dst.hostData[dstIndex] = (src.hostData[srcIndex]) / 255.f;
        dst.hostData[dstIndex + size] = (src.hostData[srcIndex + 1]) / 255.f;
        dst.hostData[dstIndex + 2 * size] = (src.hostData[srcIndex + 2]) / 255.f;
    }
    else
    {
        dst.hostData[dstIndex] = 0;
        dst.hostData[dstIndex + size] = 0;
        dst.hostData[dstIndex + 2 * size] = 0;
    }
}

void dewarpImagePixelFilteredNormalized(const Image& src, const ImageFloat& dst, const LinearPixelFilter& linearPixelFilter, int dstIndex)
{
    int size = dst.width * dst.height;

    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        

        for (int channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + size * channelIndex;
            dst.hostData[dstChannelIndex] = (src.hostData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio) / 255.f;
            dst.hostData[dstChannelIndex] += (src.hostData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio) / 255.f;
            dst.hostData[dstChannelIndex] += (src.hostData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio) / 255.f;
            dst.hostData[dstChannelIndex] += (src.hostData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio) / 255.f;
        }
    }
    else
    {
        dst.hostData[dstIndex] = 0;
        dst.hostData[dstIndex + size] = 0;
        dst.hostData[dstIndex + 2 * size] = 0;
    }
}

int calculateOffset(const Dim2<int>& dim, float aspectRatio)
{
    return ((dim.height - int(dim.height * aspectRatio)) / 2) * dim.width;
}

}

CpuDarknetFisheyeDewarper::CpuDarknetFisheyeDewarper(float outputAspectRatio)
    : outputAspectRatio_(outputAspectRatio)
{
}

void CpuDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const
{
    int size = dst.height * dst.width;
    int offset = calculateOffset(dst, outputAspectRatio_);
    
    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        int srcIndex = calculateSourcePixelIndex(srcPosition, src);
        dewarpImagePixelNormalized(src, dst, srcIndex, index + offset);
    }
}

void CpuDarknetFisheyeDewarper::dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;
    int offset = calculateOffset(dst, outputAspectRatio_);

    for (int index = 0; index < size; ++index)
    {
        dewarpImagePixelNormalized(src, dst, mapping.hostData[index], index + offset);
    }
}

void CpuDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const
{
    int size = dst.height * dst.width;
    int offset = calculateOffset(dst, outputAspectRatio_);
    
    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilter(srcPosition, src);
        dewarpImagePixelFilteredNormalized(src, dst, linearPixelFilter, index + offset);
    }
}

void CpuDarknetFisheyeDewarper::dewarpImageFiltered(const Image& src, const ImageFloat& dst, const FilteredDewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;
    int offset = calculateOffset(dst, outputAspectRatio_);
        
    for (int index = 0; index < size; ++index)
    {
        dewarpImagePixelFilteredNormalized(src, dst, mapping.hostData[index], index + offset);
    }
}

void CpuDarknetFisheyeDewarper::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                    const DewarpingMapping& mapping) const
{
    mappingFiller_.fillDewarpingMapping(src, params, mapping);
}

void CpuDarknetFisheyeDewarper::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                             const FilteredDewarpingMapping& mapping) const
{
    mappingFiller_.fillFilteredDewarpingMapping(src, params, mapping);
}

void CpuDarknetFisheyeDewarper::prepareOutputImage(ImageFloat& dst) const
{
    fillArray(dst.hostData, 0.5f, dst.size);
}

Dim2<int> CpuDarknetFisheyeDewarper::getRectifiedOutputDim(const Dim2<int>& dst) const
{
    return Dim2<int>(dst.width, dst.height * outputAspectRatio_);
}
} // Model
