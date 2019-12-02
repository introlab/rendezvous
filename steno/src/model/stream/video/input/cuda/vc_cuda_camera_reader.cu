#include "vc_cuda_camera_reader.h"

#include "model/stream/utils/cuda_utils.cuh"

#include <cstring>

namespace Model
{
VcCudaCameraReader::VcCudaCameraReader(std::shared_ptr<VideoConfig> videoConfig)
    : VcCameraReader(videoConfig, 3)
    , nextImage_(nullptr)
    , pageLockedImage_(videoConfig->resolution.width, videoConfig->resolution.height, videoConfig->imageFormat)
{
    checkCuda(cudaMallocHost(&pageLockedImage_.hostData, pageLockedImage_.size, 0));

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.allocateObject(images_.current());
        images_.next();
    }

    checkCuda(cudaStreamCreate(&stream_));
}

VcCudaCameraReader::~VcCudaCameraReader()
{
    cudaFreeHost(pageLockedImage_.hostData);

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.deallocateObject(images_.current());
        images_.next();
    }

    cudaStreamDestroy(stream_);
}

void VcCudaCameraReader::open()
{
    BaseCameraReader::open();
    nextImage_ = &VcCameraReader::readImage();
    copyImageToDevice(*nextImage_);
    cudaStreamSynchronize(stream_);
}

void VcCudaCameraReader::close()
{
    BaseCameraReader::close();
    nextImage_ = nullptr;
}

const Image& VcCudaCameraReader::readImage()
{
    if (nextImage_ == nullptr)
    {
        throw std::runtime_error("Camera reader frame is null, this is not supposed to occur!");
    }

    const Image& image = *nextImage_;
    nextImage_ = &VcCameraReader::readImage();
    copyImageToDevice(*nextImage_);

    return image;
}

void VcCudaCameraReader::copyImageToDevice(const Image& image)
{
    // Copy the image data to a page-locked image (this is for faster async copy to device memory)
    std::memcpy(pageLockedImage_.hostData, image.hostData, image.size);

    // Wait for the previous async copy completion
    cudaStreamSynchronize(stream_);

    // Async copy to device memory
    cudaMemcpyAsync(image.deviceData, pageLockedImage_.hostData, image.size, cudaMemcpyHostToDevice, stream_);
}
}    // namespace Model