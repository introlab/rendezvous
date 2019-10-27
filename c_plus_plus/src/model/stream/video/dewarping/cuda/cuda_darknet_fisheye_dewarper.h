#ifndef CUDA_DARKNET_FISHEYE_DEWARPER_H
#define CUDA_DARKNET_FISHEYE_DEWARPER_H

#include <cuda_runtime.h>

#include "model/stream/video/dewarping/cuda/cuda_dewarping_mapping_filler.h"
#include "model/stream/video/dewarping/i_detection_fisheye_dewarper.h"

namespace Model
{
class CudaDarknetFisheyeDewarper : public IDetectionFisheyeDewarper
{
   public:
    CudaDarknetFisheyeDewarper(cudaStream_t stream, float outputAspectRatio);

    void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const override;
    void dewarpImage(const Image& src, const ImageFloat& dst, const DewarpingMapping& mapping) const override;
    void dewarpImageFiltered(const Image& src, const ImageFloat& dst, const DewarpingParameters& params) const override;
    void dewarpImageFiltered(const Image& src, const ImageFloat& dst,
                             const FilteredDewarpingMapping& mapping) const override;
    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                              const DewarpingMapping& mapping) const override;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params,
                                      const FilteredDewarpingMapping& mapping) const override;
    void prepareOutputImage(ImageFloat& dst) const override;
    Dim2<int> getRectifiedOutputDim(const Dim2<int>& dst) const override;

   private:
    CudaDewarpingMappingFiller mappingFiller_;
    float outputAspectRatio_;
    cudaStream_t stream_;
};

}    // namespace Model

#endif    // !CUDA_DARKNET_FISHEYE_DEWARPER_H
