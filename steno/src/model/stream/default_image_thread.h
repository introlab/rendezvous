#ifndef DEFAULT_IMAGE_THREAD_H
#define DEFAULT_IMAGE_THREAD_H

#include <memory>

#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"
#include "model/utils/observer/i_observer.h"
#include "model/utils/observer/subject.h"

namespace Model
{
class DefaultImageThread : public Thread, public Subject
{
   public:
    DefaultImageThread(std::shared_ptr<IVideoOutput> videoOutput, std::shared_ptr<VideoConfig> videoConfig);

    enum class ThreadStatus
    {
        RUNNING,
        STOPPED,
        CRASHED
    };

    ThreadStatus getState()
    {
        return m_state;
    }

   protected:
    void run() override;

   private:
    std::shared_ptr<IVideoOutput> m_videoOutput;
    std::shared_ptr<VideoConfig> m_videoConfig;
    ThreadStatus m_state = ThreadStatus::STOPPED;
};

}    // namespace Model

#endif    // DEFAULT_IMAGE_THREAD_H
