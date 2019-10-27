#ifndef IMPLEMENTATION_FACTORY_H
#define IMPLEMENTATION_FACTORY_H

#include <memory>

#include "model/stream/utils/alloc/i_object_factory.h"
#include "model/stream/utils/images/i_image_converter.h"
#include "model/stream/utils/threads/sync/i_synchronizer.h"
#include "model/stream/video/detection/i_detector.h"
#include "model/stream/video/dewarping/i_detection_fisheye_dewarper.h"
#include "model/stream/video/dewarping/i_fisheye_dewarper.h"
#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class ImplementationFactory
{
   public:
    explicit ImplementationFactory(bool useZeroCopyIfSupported);
    virtual ~ImplementationFactory();

    std::unique_ptr<IDetector> getDetector(const std::string& configFile, const std::string& weightsFile,
                                           const std::string& metaFile);
    std::unique_ptr<IObjectFactory> getObjectFactory();
    std::unique_ptr<IObjectFactory> getDetectionObjectFactory();
    std::unique_ptr<IFisheyeDewarper> getFisheyeDewarper();
    std::unique_ptr<IDetectionFisheyeDewarper> getDetectionFisheyeDewarper(float aspectRatio);
    std::unique_ptr<ISynchronizer> getSynchronizer();
    std::unique_ptr<ISynchronizer> getDetectionSynchronizer();
    std::unique_ptr<IImageConverter> getImageConverter();
    std::unique_ptr<IVideoInput> getFileImageReader(const std::string& imageFilePath, ImageFormat format);
    std::unique_ptr<IVideoInput> getCameraReader(const VideoConfig& cameraConfig);

   private:
    bool useZeroCopyIfSupported_;
    bool isZeroCopySupported_;
};

}    // namespace Model

#endif    //! IMPLEMENTATION_FACTORY_H
