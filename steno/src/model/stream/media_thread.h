#ifndef MEDIA_THREAD_H
#define MEDIA_THREAD_H

#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/i_audio_sink.h"
#include "model/stream/audio/i_audio_source.h"
#include "model/stream/audio/i_position_source.h"
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

namespace Model
{
class MediaThread : public Thread
{
   public:
    MediaThread(std::unique_ptr<IAudioSource> audioSource, std::unique_ptr<IAudioSink> audioSink,
                std::unique_ptr<IPositionSource> positionSource, std::unique_ptr<IVideoInput> videoInput,
                std::unique_ptr<IFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
                std::unique_ptr<IVideoOutput> videoOutput, std::unique_ptr<ISynchronizer> synchronizer,
                std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
                std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IImageConverter> imageConverter,
                const DewarpingConfig& dewarpingConfig, const VideoConfig& inputConfig, const VideoConfig& outputConfig,
                const AudioConfig& audioInputConfig, const AudioConfig& audioOutputConfig);

   protected:
    void run() override;

   private:
    std::unique_ptr<IAudioSource> audioSource_;
    std::unique_ptr<IAudioSink> audioSink_;
    std::unique_ptr<IPositionSource> positionSource_;
    std::unique_ptr<IVideoInput> videoInput_;
    std::unique_ptr<IFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<IVideoOutput> videoOutput_;
    std::unique_ptr<ISynchronizer> synchronizer_;
    std::unique_ptr<VirtualCameraManager> virtualCameraManager_;
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    std::unique_ptr<IImageConverter> imageConverter_;

    const DewarpingConfig& dewarpingConfig_;
    const VideoConfig& videoInputConfig_;
    const VideoConfig& videoOutputConfig_;
    const AudioConfig& audioInputConfig_;
    const AudioConfig& audioOutputConfig_;
};

}    // namespace Model

#endif    //! MEDIA_THREAD_H
