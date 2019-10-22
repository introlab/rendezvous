#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include <string>

#include "utils/models/Dim2.h"
#include "utils/images/ImageFormat.h"

struct CameraConfig
{

    CameraConfig() = default;
    CameraConfig(int width, int height, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat);
    CameraConfig(const Dim2<int>& resolution, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat);

    Dim2<int> resolution;
    int fpsTarget;
    std::string deviceName;
    ImageFormat imageFormat;

};

#endif // !CAMERA_CONFIG_H
