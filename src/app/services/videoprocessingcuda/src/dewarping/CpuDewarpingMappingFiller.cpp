#include "CpuDewarpingMappingFiller.h"

#include "dewarping/DewarpingHelper.h"

void CpuDewarpingMappingFiller::fillDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, 
                                                     const DewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = dewarping::calculateSourcePixelPosition(mapping, params, index);
        mapping.hostData[index] = dewarping::calculateSourcePixelIndex(srcPosition, src);
    }
}

void CpuDewarpingMappingFiller::fillFilteredDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, 
                                                             const FilteredDewarpingMapping& mapping) const
{
    int size = mapping.height * mapping.width;

    for (int index = 0; index < size; ++index)
    {
        Point<float> srcPosition = dewarping::calculateSourcePixelPosition(mapping, params, index);
        mapping.hostData[index] = dewarping::calculateLinearPixelFilter(srcPosition, src);
    }
}