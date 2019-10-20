#ifndef CUDA_SYNCHRONIZER_H
#define CUDA_SYNCHRONIZER_H

#include <cuda_runtime.h>

#include "ISynchronizer.h"

class CudaSynchronizer : public ISynchronizer
{
public:

    CudaSynchronizer(cudaStream_t stream);
    void sync() const override;

private:

    cudaStream_t stream_;

};

#endif //!CUDA_SYNCHRONIZER_H