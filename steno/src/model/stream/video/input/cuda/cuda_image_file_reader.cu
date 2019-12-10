#include "cuda_image_file_reader.h"

namespace Model
{
CudaImageFileReader::CudaImageFileReader(const std::string& imageFilePath, ImageFormat format)
    : ImageFileReader(imageFilePath, format)
{
    deviceCudaObjectFactory_.allocateObject(image_);
    cudaMemcpy(image_.deviceData, image_.hostData, image_.size, cudaMemcpyHostToDevice);
}

CudaImageFileReader::~CudaImageFileReader()
{
    deviceCudaObjectFactory_.deallocateObject(image_);
}

bool CudaImageFileReader::readImage(Image& image)
{
    return ImageFileReader::readImage(image);
}
}    // namespace Model