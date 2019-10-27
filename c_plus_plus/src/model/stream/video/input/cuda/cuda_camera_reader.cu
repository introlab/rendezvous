#include "cuda_camera_reader.h"

#include "model/stream/utils/cuda_utils.cuh"

namespace Model
{
CudaCameraReader::CudaCameraReader(const VideoConfig& videoConfig)
    : CameraReader(videoConfig, 3)
    , nextImage_(nullptr)
{
    const std::unique_ptr<IndexedImage[]>& buffers = images_.buffers();
    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.allocateObject(buffers[i].image);
    }

    checkCuda(cudaStreamCreate(&stream_));

    // Read a frame to prepare the next read
    nextImage_ = &CameraReader::readImage();
    cudaMemcpyAsync(nextImage_->deviceData, nextImage_->hostData, nextImage_->size, cudaMemcpyHostToDevice, stream_);
}

CudaCameraReader::~CudaCameraReader()
{
    const std::unique_ptr<IndexedImage[]>& buffers = images_.buffers();
    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        deviceCudaObjectFactory_.deallocateObject(buffers[i].image);
    }

    cudaStreamDestroy(stream_);
}

const Image& CudaCameraReader::readImage()
{
    if (nextImage_ == nullptr)
    {
        throw std::runtime_error("Camera reader frame is null, this is not supposed to occur!");
    }

    const Image& image = *nextImage_;
    cudaStreamSynchronize(stream_);

    nextImage_ = &CameraReader::readImage();
    cudaMemcpyAsync(nextImage_->deviceData, nextImage_->hostData, nextImage_->size, cudaMemcpyHostToDevice, stream_);

    return image;
}
}    // namespace Model