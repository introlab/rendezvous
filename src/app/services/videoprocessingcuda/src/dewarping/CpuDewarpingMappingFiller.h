#ifndef CPU_DEWARPING_MAPPING_FILLER_H
#define CPU_DEWARPING_MAPPING_FILLER_H

#include "dewarping/models/DewarpingMapping.h"
#include "dewarping/models/DewarpingParameters.h"
#include "utils/models/Dim3.h"

class CpuDewarpingMappingFiller
{
public:

    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const;

};

#endif // !CPU_DEWARPING_MAPPING_FILLER_H