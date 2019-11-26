#ifndef STREAM_H
#define STREAM_H

#include "model/stream/i_stream.h"

#include "model/config/config.h"
#include "model/stream/audio/odas/odas_client.h"
#include "model/stream/media_thread.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/video/detection/detection_thread.h"
#include "model/stream/video/impl/implementation_factory.h"

#include <memory>

#include <QState>

namespace Model
{
class Stream : public IStream, public IObserver
{
    Q_OBJECT
   public:
    Stream(std::shared_ptr<Config> config);
    ~Stream() override;

    void start() override;
    void stop() override;
    void join() override;
    IStream::State state() const override
    {
        return m_state;
    }

    void updateObserver() override;

   private:
    void updateState(const IStream::State& state);

    IStream::State m_state;

    std::unique_ptr<MediaThread> m_mediaThread;
    std::unique_ptr<DetectionThread> m_detectionThread;
    std::unique_ptr<OdasClient> m_odasClient;
    std::unique_ptr<IObjectFactory> m_objectFactory;
    std::shared_ptr<LockTripleBuffer<Image>> m_imageBuffer;
    std::shared_ptr<Config> m_config;
    ImplementationFactory m_implementationFactory;
};

}    // namespace Model

#endif    //! STREAM_H
