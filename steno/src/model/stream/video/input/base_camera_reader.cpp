#include "base_camera_reader.h"

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


BaseCameraReader::BaseCameraReader(std::shared_ptr<VideoConfig> videoConfig)
    : videoConfig_(videoConfig)
    , buffer_({})
    , fd_(-1)
{
    buffer_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
}

void BaseCameraReader::open()
{
    fd_ = ::open(videoConfig_->deviceName.c_str(), O_RDWR);

    if (fd_ == ERROR_CODE)
    {
        throw std::runtime_error("Could not open camera " + videoConfig_->deviceName);
    }

    checkCaps();
    setImageFormat();
    initializeInternal();

    if (xioctl(VIDIOC_STREAMON, &buffer_.type) == ERROR_CODE)
    {
        throw std::runtime_error("Failed to start camera capture");
    }
}

void BaseCameraReader::close()
{
    if (xioctl(VIDIOC_STREAMOFF, &buffer_.type) == ERROR_CODE)
    {
        std::cerr << "Failed to stop camera capture" << std::endl;
        return;
    }

    finalizeInternal();

    ::close(fd_);
    fd_ = -1;
}

void BaseCameraReader::checkCaps()
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

void BaseCameraReader::setImageFormat()
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

int BaseCameraReader::xioctl(int request, void* arg)
{
    int result;

    do
    {
        result = ioctl(fd_, request, arg);
    } while (result == -1 && EINTR == errno);    // Try until it succeeds or an error occur

    errno = 0;

    return result;
}
}    // namespace Model
