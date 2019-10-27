#include "cuda_synchronizer.h"

CudaSynchronizer::CudaSynchronizer(cudaStream_t stream)
    : stream_(stream)
{
}

void CudaSynchronizer::sync() const
{
    cudaStreamSynchronize(stream_);
}