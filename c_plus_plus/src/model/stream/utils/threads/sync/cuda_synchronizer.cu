#include "cuda_synchronizer.h"

namespace Model
{
CudaSynchronizer::CudaSynchronizer(cudaStream_t stream)
    : stream_(stream)
{
}

void CudaSynchronizer::sync() const
{
    cudaStreamSynchronize(stream_);
}

}    // namespace Model