#ifndef CUDA_CAMERA_READER_H
#define CUDA_CAMERA_READER_H

#include <cuda_runtime.h>

#include "model/stream/utils/alloc/cuda/device_cuda_object_factory.h"
#include "model/stream/video/input/camera_reader.h"

namespace Model
{
class CudaCameraReader : public CameraReader
{
   public:
    explicit CudaCameraReader(const VideoConfig& videoConfig);
    virtual ~CudaCameraReader();

    const Image& readImage() override;

   private:
    DeviceCudaObjectFactory deviceCudaObjectFactory_;
    const Image* nextImage_;
    cudaStream_t stream_;
};

}    // namespace Model

#endif    //! CUDA_CAMERA_READER_H
