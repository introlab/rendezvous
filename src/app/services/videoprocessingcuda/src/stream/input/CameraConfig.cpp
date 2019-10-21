#include "CameraConfig.h"

CameraConfig::CameraConfig(int width, int height, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat)
    : resolution(width, height)
    , fpsTarget(fpsTarget)
    , deviceName(deviceName)
    , imageFormat(imageFormat)
{
}

CameraConfig::CameraConfig(const Dim2<int>& resolution, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat)
    : resolution(resolution)
    , fpsTarget(fpsTarget)
    , deviceName(deviceName)
    , imageFormat(imageFormat)
{
}