#ifndef CPU_FISHEYE_DEWARPER_H
#define CPU_FISHEYE_DEWARPER_H

#include "dewarping/CpuDewarpingMappingFiller.h"
#include "IFisheyeDewarper.h"

class CpuFisheyeDewarper : public IFisheyeDewarper
{
public:

    void dewarpImage(const Image& src, const Image& dst, const DewarpingParameters& params) const override;
    void dewarpImage(const Image& src, const Image& dst, const DewarpingMapping& mapping) const override;
    void dewarpImageFiltered(const Image& src, const Image& dst, const DewarpingParameters& params) const override;
    void dewarpImageFiltered(const Image& src, const Image& dst, const FilteredDewarpingMapping& mapping) const override;
    void fillDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const override;
    void fillFilteredDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const override;

private:

    CpuDewarpingMappingFiller mappingFiller_;
    
};

#endif // !CPU_FISHEYE_DEWARPER_H