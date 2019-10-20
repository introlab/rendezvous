#ifndef STREAM_H
#define STREAM_H

#include <memory>

#include "streaming/input/CameraConfig.h"
#include "detection/DetectionThread.h"
#include "dewarping/models/DewarpingConfig.h"
#include "VideoThread.h"

class Stream
{
public:

    Stream(const CameraConfig& cameraConfig, const DewarpingConfig& dewarpingConfig);

    void start();
    void stop();

private:

    CameraConfig cameraConfig_;
    DewarpingConfig dewarpingConfig_;

    std::unique_ptr<VideoThread> videoThread_;
    std::unique_ptr<DetectionThread> detectionThread_;
};

#endif //!STREAM_H
