#include "CpuDewarpingMappingFiller.h"

#include "dewarping/DewarpingHelper.h"

void CpuDewarpingMappingFiller::fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                     const DewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(mapping, params, index);
        mapping.hostData[index] = calculateSourcePixelIndex(srcPosition, src);
    }
}

void CpuDewarpingMappingFiller::fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, 
                                                             const FilteredDewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = calculateSourcePixelPosition(mapping, params, index);
        mapping.hostData[index] = calculateLinearPixelFilter(srcPosition, src);
    }
}