#ifndef VIDEO_CONFIG_H
#define VIDEO_CONFIG_H

#include <string>

#include "utils/models/Dim2.h"
#include "utils/images/ImageFormat.h"

struct VideoConfig
{

    VideoConfig() = default;
    VideoConfig(int width, int height, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat)
        : resolution(width, height)
        , fpsTarget(fpsTarget)
        , deviceName(deviceName)
        , imageFormat(imageFormat)
    {
    }

    VideoConfig(const Dim2<int>& resolution, int fpsTarget, const std::string& deviceName, ImageFormat imageFormat)
        : resolution(resolution)
        , fpsTarget(fpsTarget)
        , deviceName(deviceName)
        , imageFormat(imageFormat)
    {
    }

    Dim2<int> resolution;
    int fpsTarget;
    std::string deviceName;
    ImageFormat imageFormat;

};

#endif // !VIDEO_CONFIG_H
