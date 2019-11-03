#ifndef STREAM_H
#define STREAM_H

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

#include <memory>

#include <QState>
#include <QStateMachine>

namespace Model
{
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
    IStream::State state() const override { return m_state; }

    void updateObserver() override;

   private:
    void updateState(const IStream::State &state);

    IStream::State m_state;

    VideoConfig m_videoInputConfig;
    VideoConfig m_videoOutputConfig;
    AudioConfig m_audioInputConfig;
    AudioConfig m_audioOutputConfig;
    DewarpingConfig m_dewarpingConfig;

    std::unique_ptr<MediaThread> m_mediaThread;
    std::unique_ptr<DetectionThread> m_detectionThread;
    std::unique_ptr<OdasClient> m_odasClient;
    std::unique_ptr<IObjectFactory> m_objectFactory;
    std::shared_ptr<LockTripleBuffer<Image>> m_imageBuffer;
    ImplementationFactory m_implementationFactory;
};

}    // namespace Model

#endif    //! STREAM_H
