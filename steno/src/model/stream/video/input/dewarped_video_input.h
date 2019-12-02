#ifndef DEWARPED_VIDEO_INPUT_H
#define DEWARPED_VIDEO_INPUT_H

#include "model/config/config.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/audio/i_position_source.h"
#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/utils/images/i_image_converter.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/utils/threads/sync/i_synchronizer.h"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/detection/detection_thread.h"
#include "model/stream/video/dewarping/i_fisheye_dewarper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/video_config.h"
#include "model/stream/video/virtualcamera/virtual_camera_manager.h"

namespace Model
{
class DewarpedVideoInput : public IVideoInput, protected Thread
{
public:

    DewarpedVideoInput(std::unique_ptr<IVideoInput> videoInput, std::unique_ptr<IFisheyeDewarper> dewarper, 
                       std::unique_ptr<IObjectFactory> objectFactory, std::unique_ptr<ISynchronizer> synchronizer,
                       std::unique_ptr<VirtualCameraManager> virtualCameraManager,
                       std::unique_ptr<DetectionThread> detectionThread,
                       std::shared_ptr<LockTripleBuffer<RGBImage>> imageBuffer, std::unique_ptr<IImageConverter> imageConverter,
                       std::shared_ptr<IPositionSource> positionSource,
                       std::shared_ptr<DewarpingConfig> dewarpingConfig, std::shared_ptr<VideoConfig> videoInputConfig,
                       std::shared_ptr<VideoConfig> videoOutputConfig,
                       int bufferCount,
                       float classifierRangeThreshold);

    void open() override;
    void close() override;
    bool readImage(Image& image) override;

protected:
    void run() override;

private:
    void queueOutputImage(const Image& image);
    void updateVirtualCameras(int frameTimeMs);
    const RGBImage& getRgbFisheyeImage();
    Image dewarpInOutputFormat(const RGBImage& rgbFisheyeImage, const SphericalAngleRect& dewarpArea,
                               const Dim2<int>& dewarpDim, const RGBImage& allocatedRgbImage, 
                               const Image& allocatedOutputImage);
    void addDewarpedImageBuffers(const Dim2<int> maxVcDim);
    void cleanDewarpedImageBuffers();

    std::unique_ptr<IVideoInput> videoInput_;
    std::unique_ptr<IFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<ISynchronizer> synchronizer_;
    std::unique_ptr<VirtualCameraManager> virtualCameraManager_;
    std::unique_ptr<DetectionThread> detectionThread_;
    std::shared_ptr<LockTripleBuffer<RGBImage>> imageBuffer_;
    std::unique_ptr<IImageConverter> imageConverter_;

    std::shared_ptr<IPositionSource> positionSource_;

    std::shared_ptr<DewarpingConfig> dewarpingConfig_;
    std::shared_ptr<VideoConfig> videoInputConfig_;
    std::shared_ptr<VideoConfig> videoOutputConfig_;

    moodycamel::ReaderWriterQueue<Image> outputImageQueue_;
    int bufferCount_;

    float classifierRangeThreshold_;

    Point<float> fisheyeCenter_;
    std::vector<RGBImage> vcRgbImages_;
    std::vector<Image> vcOutputFormatImages_;

};
}   // Model

#endif  // DEWARPED_VIDEO_INPUT_H