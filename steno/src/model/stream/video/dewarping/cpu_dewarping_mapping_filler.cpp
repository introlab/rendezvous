#include "cpu_dewarping_mapping_filler.h"

#include "model/stream/video/dewarping/dewarping_helper.h"

namespace Model
{
void CpuDewarpingMappingFiller::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                                     const DewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> normalizedPixel = getNormalizedPixelFromIndex(index, mapping);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixel(normalizedPixel, params);
        mapping.hostData[index] = getSourcePixelIndex(srcPosition, src) * 3;  // For now only can dewarp in RGB format
    }
}

void CpuDewarpingMappingFiller::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                                             const FilteredDewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> normalizedPixel = getNormalizedPixelFromIndex(index, mapping);
        Point<float> srcPosition = getSourcePixelFromDewarpedImageNormalizedPixel(normalizedPixel, params);
        mapping.hostData[index] = getLinearPixelFilter(srcPosition, src);
    }
}
}    // namespace Model
