#ifndef VIDEO_THREAD_H
#define VIDEO_THREAD_H

#include "dewarping/IFisheyeDewarper.h"
#include "dewarping/models/DewarpingConfig.h"
#include "stream/input/IVideoInput.h"
#include "stream/output/IVideoOutput.h"
#include "stream/VideoConfig.h"
#include "utils/alloc/IObjectFactory.h"
#include "utils/images/IImageConverter.h"
#include "utils/threads/LockTripleBuffer.h"
#include "utils/threads/readerwriterqueue.h"
#include "utils/threads/sync/ISynchronizer.h"
#include "utils/threads/Thread.h"
#include "virtualcamera/VirtualCameraManager.h"

class VideoThread : public Thread
{
public:

    VideoThread(std::unique_ptr<IVideoInput> videostream, std::unique_ptr<IFisheyeDewarper> dewarper,
                std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<IVideoOutput> imageConsumer,
                std::unique_ptr<ISynchronizer> synchronizer, std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue,
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
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue_;
    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    std::unique_ptr<IImageConverter> imageConverter_;
    DewarpingConfig dewarpingConfig_;
    VideoConfig inputConfig_;
    VideoConfig outputConfig_;

};

#endif //!VIDEO_PROCESSOR_THREAD_H
