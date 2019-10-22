#ifndef DETECTION_THREAD2_H
#define DETECTION_THREAD2_H

#include "detection/IDetector.h"
#include "dewarping/IDetectionFisheyeDewarper.h"
#include "dewarping/models/DewarpingConfig.h"
#include "utils/threads/LockTripleBuffer.h"
#include "utils/models/AngleRect.h"
#include "utils/images/Images.h"
#include "utils/alloc/IObjectFactory.h"
#include "utils/threads/readerwriterqueue.h"
#include "utils/threads/sync/ISynchronizer.h"
#include "utils/threads/Thread.h"

class DetectionThread : public Thread
{
public:

    DetectionThread(std::shared_ptr<LockTripleBuffer<Image>> imageBuffer, std::unique_ptr<IDetector> detector,
                     std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue,
                     std::unique_ptr<IDetectionFisheyeDewarper> dewarper, std::unique_ptr<IObjectFactory> objectFactory, 
                     std::unique_ptr<ISynchronizer> synchronizer, const DewarpingConfig& dewarpingConfig, int dewarpCount);

private:

    void run() override;

    std::vector<DewarpingParameters> getDetectionDewarpingParameters(const Dim2<int>& dim, int dewarpCount);
    std::vector<ImageFloat> getDetectionImages(const Dim2<int>& dim, int dewarpCount);
    std::vector<DewarpingMapping> getDewarpingMappings(const std::vector<DewarpingParameters>& paramsVector,
                                                       const Dim2<int>& src, const Dim2<int>& dst, int dewarpCount);

    std::shared_ptr<LockTripleBuffer<Image>> imageBuffer_;
    std::unique_ptr<IDetector> detector_;
    std::unique_ptr<IDetectionFisheyeDewarper> dewarper_;
    std::unique_ptr<IObjectFactory> objectFactory_;
    std::unique_ptr<ISynchronizer> synchronizer_;
    std::shared_ptr<moodycamel::ReaderWriterQueue<std::vector<AngleRect>>> detectionQueue_;

    DewarpingConfig dewarpingConfig_;
    int dewarpCount_;

};

#endif //!DETECTION_THREAD2_H