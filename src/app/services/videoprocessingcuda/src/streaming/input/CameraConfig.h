#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include "utils/models/Dim3.h"

struct CameraConfig
{

    CameraConfig() = default;
    CameraConfig(int width, int height, int channels, int fpsTarget);
    CameraConfig(const Dim3<int>& resolution, int fpsTarget);

    Dim3<int> resolution;
    int fpsTarget;

};

#endif // !CAMERA_CONFIG_H
