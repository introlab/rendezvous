#ifndef CUDA_SYNCHRONIZER_H
#define CUDA_SYNCHRONIZER_H

#include <cuda_runtime.h>

#include "model/stream/utils/threads/sync/i_synchronizer.h"

namespace Model
{
class CudaSynchronizer : public ISynchronizer
{
   public:
    explicit CudaSynchronizer(cudaStream_t stream);
    void sync() const override;

   private:
    cudaStream_t stream_;
};

}    // namespace Model

#endif    //! CUDA_SYNCHRONIZER_H
