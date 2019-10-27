#include "cpu_fisheye_dewarper.h"

#include "model/stream/video/dewarping/dewarping_helper.h"

namespace Model
{

namespace
{

void dewarpImagePixel(const Image& src, const Image& dst, int srcIndex, int dstIndex)
{
    if (srcIndex < int(src.size) && srcIndex > 0) // Don't need to check the other ones, as they will be ok if these are
    {
        dst.hostData[dstIndex] = src.hostData[srcIndex];
        dst.hostData[dstIndex + 1] = src.hostData[srcIndex + 1];
        dst.hostData[dstIndex + 2] = src.hostData[srcIndex + 2];
    }
    else
    {
        dst.hostData[dstIndex] = 0;
        dst.hostData[dstIndex + 1] = 0;
        dst.hostData[dstIndex + 2] = 0;
    }
}

void dewarpImagePixelFiltered(const Image& src, const Image& dst, const LinearPixelFilter& linearPixelFilter, int dstIndex)
{
    // Don't need to check the other ones, as they will be ok if these are
    if (linearPixelFilter.pc4.index < int(src.size) && linearPixelFilter.pc1.index > 0)
    {
        for (int channelIndex = 0; channelIndex < 3; ++channelIndex)
        {
            int dstChannelIndex = dstIndex + channelIndex;
            dst.hostData[dstChannelIndex] = src.hostData[linearPixelFilter.pc1.index + channelIndex] * linearPixelFilter.pc1.ratio;
            dst.hostData[dstChannelIndex] += src.hostData[linearPixelFilter.pc2.index + channelIndex] * linearPixelFilter.pc2.ratio;
            dst.hostData[dstChannelIndex] += src.hostData[linearPixelFilter.pc3.index + channelIndex] * linearPixelFilter.pc3.ratio;
            dst.hostData[dstChannelIndex] += src.hostData[linearPixelFilter.pc4.index + channelIndex] * linearPixelFilter.pc4.ratio;
        }
    }
    else
    {
        dst.hostData[dstIndex] = 0;
        dst.hostData[dstIndex + 1] = 0;
        dst.hostData[dstIndex + 2] = 0;
    }
}

}

void CpuFisheyeDewarper::dewarpImage(const Image& src, const Image& dst, const DewarpingParameters& params) const
{
    int size = dst.height * dst.width;
    
    for (int index = 0; index < size; ++index)
    {
        int dstIndex = index * 3;

        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        int srcIndex = calculateSourcePixelIndex(srcPosition, Dim3<int>(src, 3));
        dewarpImagePixel(src, dst, srcIndex, dstIndex);
    }
}

void CpuFisheyeDewarper::dewarpImage(const Image& src, const Image& dst, const DewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;
        
    for (int index = 0; index < size; ++index)
    {
        int dstIndex = index * 3;
        dewarpImagePixel(src, dst, mapping.hostData[index], dstIndex);
    }
}

void CpuFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst, const DewarpingParameters& params) const
{
    int size = dst.height * dst.width;
    
    for (int index = 0; index < size; ++index)
    {
        int dstIndex = index * 3;

        Point<float> srcPosition = calculateSourcePixelPosition(dst, params, index);
        LinearPixelFilter linearPixelFilter = calculateLinearPixelFilter(srcPosition, Dim3<int>(src, 3));
        dewarpImagePixelFiltered(src, dst, linearPixelFilter, dstIndex);
    }
}

void CpuFisheyeDewarper::dewarpImageFiltered(const Image& src, const Image& dst, const FilteredDewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;
        
    for (int index = 0; index < size; ++index)
    {
        int dstIndex = index * 3;
        dewarpImagePixelFiltered(src, dst, mapping.hostData[index], dstIndex);
    }
}

void CpuFisheyeDewarper::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const
{
    mappingFiller_.fillDewarpingMapping(src, params, mapping);
}

void CpuFisheyeDewarper::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                      const FilteredDewarpingMapping& mapping) const
{
    mappingFiller_.fillFilteredDewarpingMapping(src, params, mapping);
}
} // Model
