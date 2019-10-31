#ifndef STREAM_H
#define STREAM_H

#include <memory>

#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/odas/odas_client.h"
#include "model/stream/i_stream.h"
#include "model/stream/media_thread.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/video/detection/detection_thread.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/impl/implementation_factory.h"
#include "model/stream/video/video_config.h"
#include "model/utils/observer/i_observer.h"

namespace Model
{
enum class StreamStatus
{
    RUNNING,
    STOPPING,
    STOPPED,
    CRASHED
};

class Stream : public IStream, public IObserver
{
    Q_OBJECT
   public:
    Stream(const VideoConfig& videoInputConfig, const VideoConfig& videoOutputConfig,
           const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig,
           const DewarpingConfig& dewarpingConfig);
    ~Stream() override;

    void start() override;
    void stop() override;

    void updateObserver() override;

   private:
    VideoConfig videoInputConfig_;
    VideoConfig videoOutputConfig_;
    AudioConfig audioInputConfig_;
    AudioConfig audioOutputConfig_;
    DewarpingConfig dewarpingConfig_;

    std::unique_ptr<MediaThread> mediaThread_;
    std::unique_ptr<DetectionThread> detectionThread_;
    std::unique_ptr<OdasClient> odasClient_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    ImplementationFactory implementationFactory_;

    StreamStatus status_ = StreamStatus::STOPPED;
};

}    // namespace Model

#endif    //! STREAM_H
