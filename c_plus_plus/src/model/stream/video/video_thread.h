#ifndef VIDEO_THREAD_H
#define VIDEO_THREAD_H

#include "model/stream/video/dewarping/i_fisheye_dewarper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/utils/images/i_image_converter.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/utils/threads/sync/i_synchronizer.h"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/virtualcamera/virtual_camera_manager.h"

namespace Model
{

class VideoThread : public Thread
{
public:

    VideoThread(std::unique_ptr<IVideoInput> videoInput, std::unique_ptr<IFisheyeDewarper> dewarper,
                std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IVideoOutput> videoOutput,
                std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue,
                std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IImageConverter> imageConverter,
                const DewarpingConfig& dewarpingConfig, const VideoConfig& inputConfig, const VideoConfig& outputConfig);

protected:

    void run() override;

private:

    std::unique_ptr<IVideoInput> videoInput_;
    std::unique_ptr<IFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<IVideoOutput> videoOutput_;
    std::unique_ptr<ISynchronizer> synchronizer_;
    std::unique_ptr<VirtualCameraManager> virtualCameraManager_;
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>>> detectionQueue_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    std::unique_ptr<IImageConverter> imageConverter_;
    DewarpingConfig dewarpingConfig_;
    VideoConfig inputConfig_;
    VideoConfig outputConfig_;

};

} // Model

#endif //!VIDEO_PROCESSOR_THREAD_H

