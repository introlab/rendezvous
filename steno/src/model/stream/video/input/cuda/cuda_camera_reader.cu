#include "cuda_camera_reader.h"

#include "model/stream/utils/cuda_utils.cuh"

#include <cstring>

namespace Model
{
CudaCameraReader::CudaCameraReader(std::shared_ptr<VideoConfig> videoConfig)
    : CameraReader(videoConfig, 3)
    , nextImage_(nullptr)
    , pageLockedImage_(videoConfig->resolution.width, videoConfig->resolution.height, videoConfig->imageFormat)
{
    checkCuda(cudaMallocHost(&pageLockedImage_.hostData, pageLockedImage_.size, 0));

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.allocateObject(images_.current().image);
        images_.next();
    }

    checkCuda(cudaStreamCreate(&stream_));
}

CudaCameraReader::~CudaCameraReader()
{
    cudaFreeHost(pageLockedImage_.hostData);

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.deallocateObject(images_.current().image);
        images_.next();
    }

    cudaStreamDestroy(stream_);
}

void CudaCameraReader::open()
{
    CameraReader::open();
    nextImage_ = &CameraReader::readImage();
    copyImageToDevice(*nextImage_);
    cudaStreamSynchronize(stream_);
}

void CudaCameraReader::close()
{
    CameraReader::close();
    nextImage_ = nullptr;
}

const Image& CudaCameraReader::readImage()
{
    if (nextImage_ == nullptr)
    {
        throw std::runtime_error("Camera reader frame is null, this is not supposed to occur!");
    }

    const Image& image = *nextImage_;
    nextImage_ = &CameraReader::readImage();
    copyImageToDevice(*nextImage_);

    return image;
}

void CudaCameraReader::copyImageToDevice(const Image& image)
{
    // Copy the image data to a page-locked image (this is for faster async copy to device memory)
    std::memcpy(pageLockedImage_.hostData, image.hostData, image.size);

    // Wait for the previous async copy completion
    cudaStreamSynchronize(stream_);

    // Async copy to device memory
    cudaMemcpyAsync(image.deviceData, pageLockedImage_.hostData, image.size, cudaMemcpyHostToDevice, stream_);
}
}    // namespace Model