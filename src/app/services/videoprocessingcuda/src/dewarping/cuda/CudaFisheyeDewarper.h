#ifndef CUDA_FISHEYE_DEWARPER_H
#define CUDA_FISHEYE_DEWARPER_H

#include <cuda_runtime.h>

#include "dewarping/cuda/CudaDewarpingMappingFiller.h"
#include "dewarping/IFisheyeDewarper.h"

class CudaFisheyeDewarper : public IFisheyeDewarper
{
public:

    CudaFisheyeDewarper(const cudaStream_t& stream);

    void dewarpImage(const Image& src, const Image& dst, const DewarpingParameters& params) const override;
    void dewarpImage(const Image& src, const Image& dst, const DewarpingMapping& mapping) const override;
    void dewarpImageFiltered(const Image& src, const Image& dst, const DewarpingParameters& params) const override;
    void dewarpImageFiltered(const Image& src, const Image& dst, const FilteredDewarpingMapping& mapping) const override;
    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const override;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const override;

private:

    CudaDewarpingMappingFiller mappingFiller_;
    cudaStream_t stream_;

};

#endif // !CUDA_FISHEYE_DEWARPER_H