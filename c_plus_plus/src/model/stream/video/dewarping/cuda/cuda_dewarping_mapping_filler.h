#ifndef CUDA_DEWARPING_MAPPING_FILLER_H
#define CUDA_DEWARPING_MAPPING_FILLER_H

#include <cuda_runtime.h>

#include "model/stream/video/dewarping/models/dewarping_mapping.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/utils/models/dim2.h"

namespace Model
{

class CudaDewarpingMappingFiller
{
public:

    explicit CudaDewarpingMappingFiller(cudaStream_t stream);

    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const;

private:

    cudaStream_t stream_;

};

} // Model

#endif // !CUDA_DEWARPING_MAPPING_FILLER_H
