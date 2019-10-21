#include "CameraReader.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

namespace
{
    const int ERROR_CODE = -1;
}

int CameraReader::IndexedImage::v4l2Index = 0;

CameraReader::IndexedImage::IndexedImage(const Image& image)
    : index(v4l2Index++)
    , image(image)
{
}

CameraReader::CameraReader(const CameraConfig& cameraConfig)
        : cameraConfig_(cameraConfig)
        , images_(Image(cameraConfig.resolution.width, cameraConfig.resolution.height, cameraConfig.imageFormat))
        , buffer_({})
{
    fd_ = open(cameraConfig_.deviceName.c_str(), O_RDWR);

    if (fd_ == ERROR_CODE)
    {
        throw std::runtime_error("Could not open camera " + cameraConfig_.deviceName);
    }

    buffer_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer_.memory = V4L2_MEMORY_MMAP;

    checkCaps();
    setImageFormat();
    requestBuffers();

    // Queue a first capture for fast read
    v4l2_buffer buffer = buffer_;
    buffer.index = images_.getCurrent().index;
    queueCapture(buffer);
}

CameraReader::~CameraReader()
{
    unmapBuffer(images_.getCurrent());
    unmapBuffer(images_.getInUse1());
    unmapBuffer(images_.getInUse2());
    close(fd_);
}

const Image& CameraReader::readImage()
{
    const CameraReader::IndexedImage& currentImage = images_.getCurrent();

    // Change the getCurrent buffer
    images_.swap();
    
    // This is the buffer with the queued frame
    v4l2_buffer currentBuffer = buffer_;
    currentBuffer.index = currentImage.index;
    dequeueCapture(currentBuffer);

    // This is the buffer for the next frame
    v4l2_buffer nextBuffer = buffer_;
    nextBuffer.index = images_.getCurrent().index;
    queueCapture(nextBuffer);

    return currentImage.image;
}

void CameraReader::queueCapture(v4l2_buffer& buffer)
{
    if (xioctl(VIDIOC_QBUF, &buffer) == ERROR_CODE)
    {
        perror("Failed to querry buffer");
    }
 
    if (xioctl(VIDIOC_STREAMON, &buffer.type) == ERROR_CODE)
    {
        perror("Failed to capture image");
    }
}

void CameraReader::dequeueCapture(v4l2_buffer& buffer)
{
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    timeval tv = {};
    tv.tv_sec = 2;

    int result = select(fd_ + 1, &fds, NULL, NULL, &tv);

    if (result == ERROR_CODE)
    {
        perror("Error waiting for Frame");
    }

    if (xioctl(VIDIOC_DQBUF, &buffer) == ERROR_CODE)
    {
        perror("Failed to retrieve Frame");
    }
}
 
void CameraReader::requestBuffers()
{
    const int bufferCount = 3;

    v4l2_requestbuffers req = {};
    req.count = bufferCount;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
 
    if (xioctl(VIDIOC_REQBUFS, &req) == ERROR_CODE || req.count != bufferCount)
    {
        throw std::runtime_error("Failed to request buffers for camera " + cameraConfig_.deviceName);
    }

    mapBuffer(images_.getCurrent());
    mapBuffer(images_.getInUse1());
    mapBuffer(images_.getInUse2());
}

void CameraReader::mapBuffer(IndexedImage& indexedImage)
{
    v4l2_buffer buffer = {};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = indexedImage.index;
    
    if (xioctl(VIDIOC_QUERYBUF, &buffer) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to query buffers of camera " + cameraConfig_.deviceName);
    }

    indexedImage.image.hostData = (uint8_t*) mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);

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
    fmt.fmt.pix.width = cameraConfig_.resolution.width;
    fmt.fmt.pix.height = cameraConfig_.resolution.height;
    fmt.fmt.pix.pixelformat = getV4L2Format(cameraConfig_.imageFormat);
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(VIDIOC_S_FMT, &fmt) == ERROR_CODE)
    {
        throw std::runtime_error("Unable to set camera image format");
    }

    if (fmt.fmt.pix.pixelformat != getV4L2Format(cameraConfig_.imageFormat))
    {
        throw std::runtime_error("Camera does not support specified image format : " + getImageFormatString(cameraConfig_.imageFormat));
    }
    else if (fmt.fmt.pix.width != (unsigned int)cameraConfig_.resolution.width ||
             fmt.fmt.pix.height != (unsigned int)cameraConfig_.resolution.height)
    {
        throw std::runtime_error("Camera does not support specified image size : " + std::to_string(cameraConfig_.resolution.width)
                                 + " x " + std::to_string(cameraConfig_.resolution.height));
    }
    else
    {
        std::cout << "Camera selected format : " << getV4L2FormatString(fmt.fmt.pix.pixelformat) 
                  << " in " << fmt.fmt.pix.width << " x " << fmt.fmt.pix.height << std::endl;
    }
}

int CameraReader::xioctl(int request, void* arg)
{
    int result;

    do 
    {
        result = ioctl(fd_, request, arg);
    }
    while (result == -1 && EINTR == errno); // Try until it succeeds or an error occur

    return result;
}