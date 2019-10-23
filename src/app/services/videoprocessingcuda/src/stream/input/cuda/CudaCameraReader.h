#ifndef CUDA_CAMERA_READER_H
#define CUDA_CAMERA_READER_H

#include <cuda_runtime.h>

#include "stream/input/CameraReader.h"
#include "utils/alloc/cuda/DeviceCudaObjectFactory.h"

class CudaCameraReader : public CameraReader
{
public:

    CudaCameraReader(const VideoConfig& videoConfig);
    virtual ~CudaCameraReader();

    const Image& readImage() override;

private:

    DeviceCudaObjectFactory deviceCudaObjectFactory_;
    const Image* nextImage_;
    cudaStream_t stream_;

};

#endif //!CUDA_CAMERA_READER_H