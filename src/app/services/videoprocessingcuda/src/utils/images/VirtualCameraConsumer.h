#ifndef VIRTUAL_CAMERA_CONSUMER_H
#define VIRTUAL_CAMERA_CONSUMER_H

#include <string>

#include "utils/images/IImageConsumer.h"
#include "streaming/out/VirtualCameraDevice.h"

class VirtualCameraConsumer : public IImageConsumer
{
public:

    VirtualCameraConsumer(const char* virtualCameraName, int virtualCameraWidth, int virtualCameraHeight, int virtualCameraFPS);
    virtual ~VirtualCameraConsumer();

    void consumeImage(const Image& image) override;

private:
    VirtualCameraDevice virtualCameraDevice_;

};

#endif // !VIRTUAL_CAMERA_CONSUMER_H
