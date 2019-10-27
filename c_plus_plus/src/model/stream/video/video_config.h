#ifndef VIDEO_CONFIG_H
#define VIDEO_CONFIG_H

#include <string>

#include "model/stream/utils/models/dim2.h"
#include "model/stream/utils/images/image_format.h"

namespace Model
{

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

} // Model

#endif // !VIDEO_CONFIG_H

