#ifndef VC_CUDA_CAMERA_READER_H
#define VC_CUDA_CAMERA_READER_H

#include <cuda_runtime.h>

#include "model/stream/utils/alloc/cuda/device_cuda_object_factory.h"
#include "model/stream/video/input/vc_camera_reader.h"

namespace Model
{
class VcCudaCameraReader : public VcCameraReader
{
   public:
    explicit VcCudaCameraReader(std::shared_ptr<VideoConfig> videoConfig);
    virtual ~VcCudaCameraReader();

    void open() override;
    void close() override;
    bool readImage(Image& image) override;

   private:
    void copyImageToDevice(const Image& image);

    DeviceCudaObjectFactory deviceCudaObjectFactory_;
    Image nextImage_;
    cudaStream_t stream_;
    Image pageLockedImage_;
};

}    // namespace Model

#endif    //! VC_CUDA_CAMERA_READER_H
