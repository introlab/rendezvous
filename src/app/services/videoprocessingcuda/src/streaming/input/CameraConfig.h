#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include <string>

#include "utils/models/Dim3.h"

struct CameraConfig
{

    CameraConfig() = default;
    CameraConfig(int width, int height, int channels, int fpsTarget, const std::string& deviceName);
    CameraConfig(const Dim3<int>& resolution, int fpsTarget, const std::string& deviceName);

    Dim3<int> resolution;
    int fpsTarget;
    std::string deviceName;

};

#endif // !CAMERA_CONFIG_H
