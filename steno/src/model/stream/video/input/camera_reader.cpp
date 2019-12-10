#include "camera_reader.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

#include "model/stream/utils/time/time_utils.h"

namespace Model
{
namespace
{
const int ERROR_CODE = -1;
}

CameraReader::IndexedImage::IndexedImage(const Image& image)
    : index(0)
    , image(image)
{
}

CameraReader::CameraReader(std::shared_ptr<VideoConfig> videoConfig, std::size_t bufferCount)
    : BaseCameraReader(videoConfig)
    , images_(bufferCount,
              Image(videoConfig->resolution.width, videoConfig->resolution.height, videoConfig->imageFormat))
{
    buffer_.memory = V4L2_MEMORY_MMAP;
}

void CameraReader::open()
{
    BaseCameraReader::open();

    // Queue a first capture for fast read
    queueCapture(images_.current());
}

bool CameraReader::readImage(Image& image)
{
    CameraReader::IndexedImage& currentIndexedImage = images_.current();

    // Change the current() indexed image
    images_.next();

    // Recover the last frame in queue
    dequeueCapture(currentIndexedImage);

    // Queue the next frame
    queueCapture(images_.current());

    image = currentIndexedImage.image;

    return true;
}

void CameraReader::initializeInternal()
{
    requestBuffers(images_.size());
}

void CameraReader::finalizeInternal()
{
    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        unmapBuffer(images_.current());
        images_.next();
    }
}

void CameraReader::queueCapture(IndexedImage& indexedImage)
{
    v4l2_buffer buffer = buffer_;
    buffer.index = indexedImage.index;

    if (xioctl(VIDIOC_QBUF, &buffer) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to querry camera buffer");
    }
}

void CameraReader::dequeueCapture(IndexedImage& indexedImage)
{
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    timeval tv = {};
    tv.tv_sec = 2;

    int result = select(fd_ + 1, &fds, NULL, NULL, &tv);

    if (result == ERROR_CODE)
    {
        throw std::runtime_error("Error waiting for camera frame");
    }

    v4l2_buffer buffer = buffer_;
    buffer.index = indexedImage.index;

    if (xioctl(VIDIOC_DQBUF, &buffer) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to retrieve camera frame");
    }

    // Set capture time + timestamp correction offset
    indexedImage.image.timeStamp = systemTimeSinceEpoch() + 40000; // TODO: config
}

void CameraReader::requestBuffers(std::size_t bufferCount)
{
    v4l2_requestbuffers req = {};
    req.count = bufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(VIDIOC_REQBUFS, &req) == ERROR_CODE || req.count != bufferCount)
    {
        throw std::runtime_error("Failed to request buffers for camera " + videoConfig_->deviceName);
    }

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        IndexedImage& indexedImage = images_.current();
        indexedImage.index = i;
        mapBuffer(indexedImage);
        images_.next();
    }
}

void CameraReader::mapBuffer(IndexedImage& indexedImage)
{
    v4l2_buffer buffer = {};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = indexedImage.index;

    if (xioctl(VIDIOC_QUERYBUF, &buffer) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to query buffers of camera " + videoConfig_->deviceName);
    }

    indexedImage.image.hostData =
        (uint8_t*)mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);

    if (indexedImage.image.hostData == nullptr)
    {
        throw std::runtime_error("Camera allocated buffer is null");
    }
}

void CameraReader::unmapBuffer(IndexedImage& indexedImage)
{
    if (indexedImage.image.hostData != nullptr)
    {
        munmap(indexedImage.image.hostData, indexedImage.image.size);
        indexedImage.image.hostData = nullptr;
    }
}

}    // namespace Model
