#include "CameraConfig.h"

CameraConfig::CameraConfig(int width, int height, int channels, int fpsTarget)
    : resolution(width, height, channels)
    , fpsTarget(fpsTarget)
{
}

CameraConfig::CameraConfig(const Dim3<int>& resolution, int fpsTarget)
    : resolution(resolution)
    , fpsTarget(fpsTarget)
{
}