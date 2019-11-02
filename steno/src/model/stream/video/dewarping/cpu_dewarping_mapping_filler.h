#ifndef CPU_DEWARPING_MAPPING_FILLER_H
#define CPU_DEWARPING_MAPPING_FILLER_H

#include "model/stream/utils/models/dim3.h"
#include "model/stream/video/dewarping/models/dewarping_mapping.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"

namespace Model
{
class CpuDewarpingMappingFiller
{
   public:
    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                              const DewarpingMapping& mapping) const;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                      const FilteredDewarpingMapping& mapping) const;
};

}    // namespace Model

#endif    // !CPU_DEWARPING_MAPPING_FILLER_H
