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
    : videoConfig_(videoConfig)
    , images_(bufferCount,
              Image(videoConfig->resolution.width, videoConfig->resolution.height, videoConfig->imageFormat))
    , buffer_({})
    , fd_(-1)
{
    buffer_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer_.memory = V4L2_MEMORY_MMAP;
}

void CameraReader::open()
{
    fd_ = ::open(videoConfig_->deviceName.c_str(), O_RDWR);

    if (fd_ == ERROR_CODE)
    {
        throw std::runtime_error("Could not open camera " + videoConfig_->deviceName);
    }

    checkCaps();
    setImageFormat();
    requestBuffers(images_.size());

    if (xioctl(VIDIOC_STREAMON, &buffer_.type) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to start camera capture");
    }

    // Queue a first capture for fast read
    queueCapture(images_.current());
}

void CameraReader::close()
{
    if (xioctl(VIDIOC_STREAMOFF, &buffer_.type) == ERROR_CODE)
    {
        std::cerr << "Failed to stop camera capture" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < images_.size(); ++i)
    {
        unmapBuffer(images_.current());
        images_.next();
    }

    ::close(fd_);
    fd_ = -1;
}

const Image& CameraReader::readImage()
{
    const CameraReader::IndexedImage& currentIndexedImage = images_.current();

    // Change the current() indexed image
    images_.next();

    // Recover the last frame in queue
    dequeueCapture(currentIndexedImage);

    // Queue the next frame
    queueCapture(images_.current());

    return currentIndexedImage.image;
}

void CameraReader::queueCapture(IndexedImage& indexedImage)
{
    v4l2_buffer buffer = buffer_;
    buffer.index = indexedImage.index;

    if (xioctl(VIDIOC_QBUF, &buffer) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to querry camera buffer");
    }

    indexedImage.image.timeStamp = systemTimeSinceEpoch();
}

void CameraReader::dequeueCapture(const IndexedImage& indexedImage)
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

void CameraReader::checkCaps()
{
    v4l2_capability caps = {};

    if (xioctl(VIDIOC_QUERYCAP, &caps) == ERROR_CODE)
    {
        throw std::runtime_error("Unable to query camera capabilities");
    }

    if (!(caps.capabilities && (V4L2_CAP_STREAMING | V4L2_CAP_EXT_PIX_FORMAT)))
    {
        throw std::runtime_error("Camera is missing required capabilities");
    }

    std::cout << "Camera in use: " << caps.card << std::endl;

    v4l2_fmtdesc fmtdesc = {};
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    std::cout << "Camera supported formats : ";

    while (xioctl(VIDIOC_ENUM_FMT, &fmtdesc) == 0)
    {
        std::cout << getV4L2FormatString(fmtdesc.pixelformat) << " ";
        fmtdesc.index++;
    }

    std::cout << std::endl;
}

void CameraReader::setImageFormat()
{
    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = videoConfig_->resolution.width;
    fmt.fmt.pix.height = videoConfig_->resolution.height;
    fmt.fmt.pix.pixelformat = getV4L2Format(videoConfig_->imageFormat);
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(VIDIOC_S_FMT, &fmt) == ERROR_CODE)
    {
        throw std::runtime_error("Unable to set camera image format");
    }

    if (fmt.fmt.pix.pixelformat != getV4L2Format(videoConfig_->imageFormat))
    {
        throw std::runtime_error("Camera does not support specified image format : " +
                                 getImageFormatString(videoConfig_->imageFormat));
    }
    else if (fmt.fmt.pix.width != (unsigned int)videoConfig_->resolution.width ||
             fmt.fmt.pix.height != (unsigned int)videoConfig_->resolution.height)
    {
        throw std::runtime_error(
            "Camera does not support specified image size : " + std::to_string(videoConfig_->resolution.width) + " x " +
            std::to_string(videoConfig_->resolution.height));
    }
    else
    {
        std::cout << "Camera selected format : " << getV4L2FormatString(fmt.fmt.pix.pixelformat) << " in "
                  << fmt.fmt.pix.width << " x " << fmt.fmt.pix.height << std::endl;
    }
}

int CameraReader::xioctl(int request, void* arg)
{
    int result;

    do
    {
        result = ioctl(fd_, request, arg);
    } while (result == -1 && EINTR == errno);    // Try until it succeeds or an error occur

    return result;
}
}    // namespace Model
