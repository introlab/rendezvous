#ifndef CUDA_IMAGE_FILE_READER_H
#define CUDA_IMAGE_FILE_READER_H

#include "model/stream/utils/alloc/cuda/device_cuda_object_factory.h"
#include "model/stream/video/input/image_file_reader.h"

namespace Model
{
class CudaImageFileReader : public ImageFileReader
{
   public:
    CudaImageFileReader(const std::string& imageFilePath, ImageFormat format);
    virtual ~CudaImageFileReader();

    const Image& readImage() override;

   private:
    DeviceCudaObjectFactory deviceCudaObjectFactory_;
};

}    // namespace Model

#endif    // !CUDA_IMAGE_FILE_READER_H
