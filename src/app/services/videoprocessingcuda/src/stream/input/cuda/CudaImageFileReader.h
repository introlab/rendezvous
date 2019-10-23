#ifndef CUDA_IMAGE_FILE_READER_H
#define CUDA_IMAGE_FILE_READER_H

#include "stream/input/ImageFileReader.h"
#include "utils/alloc/cuda/DeviceCudaObjectFactory.h"

class CudaImageFileReader : public ImageFileReader
{
public:

    CudaImageFileReader(const std::string& imageFilePath, ImageFormat format);
    virtual ~CudaImageFileReader();

    const Image& readImage() override;

private:

    DeviceCudaObjectFactory deviceCudaObjectFactory_;

};

#endif // !CUDA_IMAGE_FILE_READER_H
