#include "virtual_camera_output.h"

#include <stdexcept>

namespace Model
{
VirtualCameraOutput::VirtualCameraOutput(std::shared_ptr<VideoConfig> videoConfig)
    : videoConfig_(videoConfig)
    , videoOutput_(nullptr)
{
}

void VirtualCameraOutput::open()
{
    unsigned int v4l2Format = getV4L2Format(videoConfig_->imageFormat);

    V4L2DeviceParameters param(videoConfig_->deviceName.c_str(), v4l2Format, videoConfig_->resolution.width,
                               videoConfig_->resolution.height, videoConfig_->fpsTarget);
    videoOutput_ = V4l2Output::create(param, V4l2Access::IOTYPE_READWRITE);

    if (videoOutput_ == nullptr)
    {
        throw std::runtime_error("Could not open virtual camera " + videoConfig_->deviceName);
    }
}

void VirtualCameraOutput::close()
{
    if (videoOutput_ != nullptr)
    {
        videoOutput_->stop();
        delete videoOutput_;
        videoOutput_ = nullptr;
    }
}

void VirtualCameraOutput::writeImage(const Image& image)
{
    if (videoConfig_->imageFormat == image.format)
    {
        // This check doesn't seen to work as intended, for now it seems to work without the check
        // timeval timeout;
        // if (videoOutput_->isWritable(&timeout) != -1)
        {
            videoOutput_->write((char*)image.hostData, image.size);
        }
    }
    else
    {
        throw std::invalid_argument("Virtual camera expected format " +
                                    getImageFormatString(videoConfig_->imageFormat) + " but received format " +
                                    getImageFormatString(image.format));
    }
}
}    // namespace Model
