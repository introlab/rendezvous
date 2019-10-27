#ifndef STREAM_H
#define STREAM_H

#include <memory>

#include "model/stream/i_stream.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/video/video_config.h"
#include "model/stream/video/detection/detection_thread.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/video_thread.h"

namespace Model
{

class Stream : public IStream
{
public:

    Stream(const VideoConfig& inputConfig, const VideoConfig& outputConfig, const DewarpingConfig& dewarpingConfig);
    ~Stream() override;

    void start() override;
    void stop() override;

private:

    VideoConfig inputConfig_;
    VideoConfig outputConfig_;
    DewarpingConfig dewarpingConfig_;

    std::unique_ptr<VideoThread> videoThread_;
    std::unique_ptr<DetectionThread> detectionThread_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
};

} // Model

#endif //!STREAM_H
