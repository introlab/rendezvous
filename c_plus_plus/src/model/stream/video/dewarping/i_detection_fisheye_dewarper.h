#ifndef I_DETECTION_FISHEYE_DEWARPER_H
#define I_DETECTION_FISHEYE_DEWARPER_H

#include "model/stream/video/dewarping/models/dewarping_mapping.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/utils/images/images.h"

namespace Model
{

class IDetectionFisheyeDewarper
{
public:

    virtual ~IDetectionFisheyeDewarper() = default;
    virtual void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const = 0;
    virtual void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const FilteredDewarpingMapping& mapping) const = 0;
    virtual void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const = 0;
    virtual void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const = 0;
    virtual void prepareOutputImage(ImageFloat& dst) const = 0;
    virtual Dim2<int> getRectifiedOutputDim(const Dim2<int>& dst) const = 0;

};

} // Model

#endif // !I_DETECTION_FISHEYE_DEWARPER_H
