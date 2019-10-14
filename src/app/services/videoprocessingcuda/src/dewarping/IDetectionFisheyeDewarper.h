#ifndef I_DETECTION_FISHEYE_DEWARPER_H
#define I_DETECTION_FISHEYE_DEWARPER_H

#include "dewarping/models/DewarpingMapping.h"
#include "dewarping/models/DewarpingParameters.h"
#include "utils/images/Image.h"

class IDetectionFisheyeDewarper
{
public:

    virtual ~IDetectionFisheyeDewarper() {};
    virtual void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const FilteredDewarpingMapping& mapping) const = 0;
    virtual void fillDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const = 0;
    virtual void fillFilteredDewarpingMapping(const Dim3<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const = 0;
    virtual void prepareOutputImage(ImageFloat& dst) const = 0;
    virtual Dim2<int> getRectifiedOutputDim(const Dim2<int>& dst) const = 0;

};

#endif // !I_DETECTION_FISHEYE_DEWARPER_H