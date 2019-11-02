#ifndef STREAM_H
#define STREAM_H

#include "model/stream/audio/audio_config.h"
#include "model/stream/i_stream.h"
#include "model/stream/media_thread.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/video/detection/detection_thread.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/impl/implementation_factory.h"
#include "model/stream/video/video_config.h"

#include <memory>

#include <QState>
#include <QStateMachine>

namespace Model
{
class Stream : public IStream
{
   public:
    Stream(const VideoConfig& videoInputConfig, const VideoConfig& videoOutputConfig,
           const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig,
           const DewarpingConfig& dewarpingConfig);
    ~Stream() override;

    void start() override;
    void stop() override;
    IStream::State state() const override { return m_state; }

   private:
    void updateState(const IStream::State &state);

    IStream::State m_state;

    VideoConfig videoInputConfig_;
    VideoConfig videoOutputConfig_;
    AudioConfig audioInputConfig_;
    AudioConfig audioOutputConfig_;
    DewarpingConfig dewarpingConfig_;

    std::unique_ptr<MediaThread> mediaThread_;
    std::unique_ptr<DetectionThread> detectionThread_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    ImplementationFactory implementationFactory_;
};

}    // namespace Model

#endif    //! STREAM_H
