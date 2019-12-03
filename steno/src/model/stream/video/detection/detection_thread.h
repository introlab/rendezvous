#ifndef DETECTION_THREAD_H
#define DETECTION_THREAD_H

#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/utils/threads/readerwriterqueue.h"
#include "model/stream/utils/threads/sync/i_synchronizer.h"
#include "model/stream/utils/threads/thread.h"
#include "model/stream/video/detection/i_detector.h"
#include "model/stream/video/dewarping/i_detection_fisheye_dewarper.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"

#include <memory>

namespace Model
{
class DetectionThread : public Thread
{
   public:
    DetectionThread(std::shared_ptr<LockTripleBuffer<RGBImage>> imageBuffer, std::unique_ptr<IDetector> detector,
                    std::unique_ptr<IDetectionFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory,
                    std::unique_ptr<ISynchronizer> synchronizer, std::shared_ptr<DewarpingConfig> dewarpingConfig);

    bool getDetections(std::vector<SphericalAngleRect>& detections);

   private:
    void run() override;

    std::vector<DewarpingParameters> getDetectionDewarpingParameters(const Dim2<int>& dim, int dewarpCount);
    std::vector<ImageFloat> getDetectionImages(const Dim2<int>& dim, int dewarpCount);
    std::vector<DewarpingMapping> getDewarpingMappings(const std::vector<DewarpingParameters>& paramsVector,
                                                       const Dim2<int>& src, const Dim2<int>& dst, int dewarpCount);

    std::shared_ptr<LockTripleBuffer<RGBImage>> imageBuffer_;
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<IDetectionFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<ISynchronizer> synchronizer_;

    std::shared_ptr<DewarpingConfig> dewarpingConfig_;

    moodycamel::ReaderWriterQueue<std::vector<SphericalAngleRect>> detectionQueue_;
};

}    // namespace Model

#endif    //! DETECTION_THREAD_H
