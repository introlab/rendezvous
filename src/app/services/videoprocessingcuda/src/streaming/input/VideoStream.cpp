#include "VideoStream.h"
#include <cstring>
#include <iostream>

VideoStream::VideoStream(CameraConfig cameraConfig)
    : cameraReader_(cameraConfig)
{
    image_.width = cameraConfig.resolution.width;
    image_.height = cameraConfig.resolution.height;
    image_.channels = cameraConfig.resolution.channels;
    image_.size = image_.width * image_.height * image_.channels;

    cameraReader_.start();
}

VideoStream::~VideoStream()
{
    cameraReader_.stop();
}

bool VideoStream::copyFrameData(const Image& image)
{
    image_.hostData = cameraReader_.readFrame();

    if (image_.hostData && image.size == image_.size)
    {
        std::memcpy(image.hostData, image_.hostData, image_.size * sizeof(unsigned char));
    }

    return image_.hostData != nullptr;
    
    return false;
}

bool VideoStream::getResolution(Dim3<int>& dim)
{
    dim.width = image_.width;
    dim.height = image_.height;
    dim.channels = image_.channels;

    return image_.hostData != nullptr;
}