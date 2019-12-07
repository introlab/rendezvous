#ifndef DEFAULT_STREAM_H
#define DEFAULT_STREAM_H

#include "i_stream.h"
#include "model/config/config.h"
#include "model/stream/default_image_thread.h"
#include "model/utils/observer/i_observer.h"

#include <memory>

namespace Model
{
class DefaultStream : public IStream, public IObserver
{
   public:
    DefaultStream(std::shared_ptr<Config> config);
    ~DefaultStream() override;
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
    std::shared_ptr<VideoConfig> m_config = nullptr;
    std::unique_ptr<DefaultImageThread> m_defaultImageThread = nullptr;
};

}    // namespace Model

#endif    // DEFAULT_STREAM_H
