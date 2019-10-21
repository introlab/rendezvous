#ifndef VIDEO_THREAD_H
#define VIDEO_THREAD_H

#include "dewarping/IFisheyeDewarper.h"
#include "dewarping/models/DewarpingConfig.h"
#include "streaming/input/IVideoStream.h"
#include "streaming/input/CameraConfig.h"
#include "utils/images/IImageConsumer.h"
#include "utils/models/Rectangle.h"
#include "utils/objects/IObjectFactory.h"
#include "utils/threads/LockTripleBuffer.h"
#include "utils/threads/readerwriterqueue.h"
#include "utils/threads/sync/ISynchronizer.h"
#include "utils/threads/Thread.h"
#include "virtualcamera/VirtualCameraManager.h"

class VideoThread : public Thread
{
public:

    VideoThread(std::unique_ptr<IVideoStream> videostream, std::unique_ptr<IFisheyeDewarper> dewarper,
                std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IImageConsumer> imageConsumer,
                std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue,
                std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, const DewarpingConfig& dewarpingConfig,
                const CameraConfig& cameraConfig);

protected:

    void run() override;

private:

    std::unique_ptr<IVideoStream> videostream_;
    std::unique_ptr<IFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<IImageConsumer> imageConsumer_;
    std::unique_ptr<ISynchronizer> synchronizer_;
    std::unique_ptr<VirtualCameraManager> virtualCameraManager_;
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    DewarpingConfig dewarpingConfig_;
    CameraConfig cameraConfig_;

};

#endif //!VIDEO_PROCESSOR_THREAD_H
