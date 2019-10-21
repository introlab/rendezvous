#ifndef I_FISHEYE_DEWARPER_H
#define I_FISHEYE_DEWARPER_H

#include "dewarping/models/DewarpingMapping.h"
#include "dewarping/models/DewarpingParameters.h"
#include "utils/images/Image.h"

class IFisheyeDewarper
{
public:

    virtual ~IFisheyeDewarper() {};
    virtual void dewarpImage(const Image& src, const Image& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImage(const Image& src, const Image& dst, const DewarpingMapping& mapping) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const Image& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const Image& dst, const FilteredDewarpingMapping& mapping) const = 0;
    virtual void fillDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const = 0;
    virtual void fillFilteredDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const = 0;

};

#endif // !I_FISHEYE_DEWARPER_H