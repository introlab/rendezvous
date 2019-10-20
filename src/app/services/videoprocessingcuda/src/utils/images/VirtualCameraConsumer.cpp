#include "VirtualCameraConsumer.h"
#include <iostream>

VirtualCameraConsumer::VirtualCameraConsumer(const char* virtualCameraName, int virtualCameraWidth, int virtualCameraHeight, int virtualCameraFPS)
    : virtualCameraDevice_(virtualCameraName, ImageFormat::UYVY, virtualCameraWidth, virtualCameraHeight, virtualCameraFPS)
{}

VirtualCameraConsumer::~VirtualCameraConsumer()
{
    virtualCameraDevice_.stopDevice();
}

void VirtualCameraConsumer::consumeImage(const Image& image)
{
    timeval timeout;
    if(virtualCameraDevice_.isWritable(timeout))
    {
        virtualCameraDevice_.write(image.hostData, image.width, image.height, image.channels);
    }
}