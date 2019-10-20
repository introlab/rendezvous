#include "CameraConfig.h"

CameraConfig::CameraConfig(int width, int height, int channels, int fpsTarget, const std::string& deviceName)
    : resolution(width, height, channels)
    , fpsTarget(fpsTarget)
    , deviceName(deviceName)
{
}

CameraConfig::CameraConfig(const Dim3<int>& resolution, int fpsTarget, const std::string& deviceName)
    : resolution(resolution)
    , fpsTarget(fpsTarget)
    , deviceName(deviceName)
{
}