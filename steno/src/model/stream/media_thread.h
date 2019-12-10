#ifndef MEDIA_THREAD_H
#define MEDIA_THREAD_H

#include "model/config/config.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/i_audio_sink.h"
#include "model/stream/audio/i_audio_source.h"
#include "model/stream/audio/i_position_source.h"
#include "model/stream/media_synchronizer.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/utils/images/i_image_converter.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/utils/threads/sync/i_synchronizer.h"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/dewarping/i_fisheye_dewarper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"
#include "model/stream/video/virtualcamera/virtual_camera_manager.h"
#include "model/utils/observer/subject.h"

namespace Model
{
class MediaThread : public Thread, public Subject
{
   public:
    MediaThread(std::unique_ptr<IAudioSource> audioSource, std::unique_ptr<IAudioSink> audioSink,
                std::shared_ptr<IPositionSource> positionSource, std::unique_ptr<IVideoInput> videoInput,
                std::unique_ptr<IVideoOutput> videoOutput,
                std::shared_ptr<IVirtualCameraSource> virtualCameraSource,
                std::unique_ptr<MediaSynchronizer> mediaSynchronizer,
                int framePerSeconds,
                float classifierRangeThreshold);

   protected:
    void run() override;

   private:
    std::unique_ptr<IAudioSource> audioSource_;
    std::unique_ptr<IAudioSink> audioSink_;
    std::shared_ptr<IPositionSource> positionSource_;
    std::unique_ptr<IVideoInput> videoInput_;
    std::unique_ptr<IVideoOutput> videoOutput_;
    std::shared_ptr<IVirtualCameraSource> virtualCameraSource_;
    std::unique_ptr<MediaSynchronizer> mediaSynchronizer_;
    int framePerSeconds_;
    float classifierRangeThreshold_;
};

}    // namespace Model

#endif    //! MEDIA_THREAD_H
