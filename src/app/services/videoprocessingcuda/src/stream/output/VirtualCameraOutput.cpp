#include "VirtualCameraOutput.h"

#include <stdexcept>

VirtualCameraOutput::VirtualCameraOutput(const VideoConfig& videoConfig)
    : videoConfig_(videoConfig)
    , videoOutput_(nullptr)
{
    unsigned int v4l2Format = getV4L2Format(videoConfig.imageFormat);
    
    V4L2DeviceParameters param(videoConfig_.deviceName.c_str(), v4l2Format, videoConfig_.resolution.width, 
                               videoConfig_.resolution.height, videoConfig_.fpsTarget);
    videoOutput_ = V4l2Output::create(param, V4l2Access::IOTYPE_READWRITE);

    if (!videoOutput_)
    {
        throw std::runtime_error("Could not open virtual camera " + videoConfig_.deviceName);
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
    if (videoConfig_.imageFormat == image.format)
    {
        timeval timeout;
        if (videoOutput_->isWritable(&timeout) == 1)
        {
            videoOutput_->write((char*)image.hostData, image.size);
        }
    }
    else
    {
        throw std::invalid_argument("Virtual camera expected format " + getImageFormatString(videoConfig_.imageFormat) + 
                                    " but received format " + getImageFormatString(image.format));
    }
    
}