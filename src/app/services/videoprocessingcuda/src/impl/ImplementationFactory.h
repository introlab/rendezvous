#ifndef IMPLEMENTATION_FACTORY_H
#define IMPLEMENTATION_FACTORY_H

#include <memory>

#include "detection/IDetector.h"
#include "dewarping/IDetectionFisheyeDewarper.h"
#include "dewarping/IFisheyeDewarper.h"
#include "stream/VideoConfig.h"
#include "stream/input/IVideoInput.h"
#include "utils/alloc/IObjectFactory.h"
#include "utils/images/IImageConverter.h"
#include "utils/threads/sync/ISynchronizer.h"

class ImplementationFactory
{
public:

    ImplementationFactory(bool useZeroCopyIfSupported);
    virtual ~ImplementationFactory();

    std::unique_ptr<IDetector> getDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);
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

#endif //!IMPLEMENTATION_FACTORY_H