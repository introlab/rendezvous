#include "VirtualCameraOutput.h"

#include <stdexcept>

VirtualCameraOutput::VirtualCameraOutput(const std::string& videoDevice, const Dim2<int>& dim, ImageFormat format, unsigned int fps)
    : videoOutput_(nullptr)
    , format_(format)
{
    unsigned int v4l2Format = getV4L2Format(format_);
    
    V4L2DeviceParameters param(videoDevice.c_str(), v4l2Format, dim.width, dim.height, fps);
    videoOutput_ = V4l2Output::create(param, V4l2Access::IOTYPE_READWRITE);

    if (!videoOutput_)
    {
        throw std::runtime_error("Could not open virtual camera " + videoDevice);
    }
}

VirtualCameraOutput::~VirtualCameraOutput()
{
    if (videoOutput_)
    {
        videoOutput_->stop();
        videoOutput_ = nullptr;
    }
}

void VirtualCameraOutput::writeImage(const Image& image)
{
    if (format_ == image.format)
    {
        timeval timeout;
        if (videoOutput_->isWritable(&timeout) == 1)
        {
            videoOutput_->write((char*)image.hostData, image.size);
        }
    }
    else
    {
        throw std::invalid_argument("Virtual camera expected format " + getImageFormatString(format_) + 
                                    " but received format " + getImageFormatString(image.format));
    }
    
}