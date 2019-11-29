#ifndef DEFAULT_IMAGE_THREAD_H
#define DEFAULT_IMAGE_THREAD_H

#include "memory"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class DefaultImageThread : public Thread
{
   public:
    DefaultImageThread(std::shared_ptr<IVideoOutput> videoOutput, std::shared_ptr<VideoConfig> videoConfig);

   protected:
    void run() override;

   private:
    std::shared_ptr<IVideoOutput> m_videoOutput;
    std::shared_ptr<VideoConfig> m_videoConfig;
};

}    // namespace Model

#endif    // DEFAULT_IMAGE_THREAD_H
