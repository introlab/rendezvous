#ifndef CPU_DARKNET_FISHEYE_DEWARPER_H
#define CPU_DARKNET_FISHEYE_DEWARPER_H

#include "model/stream/video/dewarping/cpu_dewarping_mapping_filler.h"
#include "model/stream/video/dewarping/i_detection_fisheye_dewarper.h"

namespace Model
{

class CpuDarknetFisheyeDewarper : public IDetectionFisheyeDewarper
{
public:

    explicit CpuDarknetFisheyeDewarper(float outputAspectRatio);

    void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const override;
    void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const override;
    void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const override;
    void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const FilteredDewarpingMapping& mapping) const override;
    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const override;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const override;
    void prepareOutputImage(ImageFloat& dst) const override;
    Dim2<int> getRectifiedOutputDim(const Dim2<int>& dst) const override;

private:

    CpuDewarpingMappingFiller mappingFiller_;
    float outputAspectRatio_;

};

} // Model

#endif // !CPU_DARKNET_FISHEYE_DEWARPER_H
