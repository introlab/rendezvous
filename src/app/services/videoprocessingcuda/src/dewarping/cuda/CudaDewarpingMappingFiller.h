#ifndef CUDA_DEWARPING_MAPPING_FILLER_H
#define CUDA_DEWARPING_MAPPING_FILLER_H

#include <cuda_runtime.h>

#include "dewarping/models/DewarpingMapping.h"
#include "dewarping/models/DewarpingParameters.h"
#include "utils/models/Dim2.h"

class CudaDewarpingMappingFiller
{
public:

    CudaDewarpingMappingFiller(cudaStream_t stream);

    void fillDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const DewarpingMapping& mapping) const;
    void fillFilteredDewarpingMapping(const Dim2<int>& src, const DewarpingParameters& params, const FilteredDewarpingMapping& mapping) const;

private:

    cudaStream_t stream_;

};

#endif // !CUDA_DEWARPING_MAPPING_FILLER_H