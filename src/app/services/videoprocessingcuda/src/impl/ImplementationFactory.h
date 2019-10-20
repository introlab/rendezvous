#ifndef IMPLEMENTATION_FACTORY_H
#define IMPLEMENTATION_FACTORY_H

#include <memory>

#include "detection/IDetector.h"
#include "dewarping/IFisheyeDewarper.h"
#include "dewarping/IDetectionFisheyeDewarper.h"
#include "utils/objects/IObjectFactory.h"
#include "utils/threads/sync/ISynchronizer.h"

class ImplementationFactory
{
public:

    ImplementationFactory(bool useZeroCopyIfSupported);

    std::unique_ptr<IDetector> getDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);
    std::unique_ptr<IObjectFactory> getObjectFactory();
    std::unique_ptr<IObjectFactory> getDetectionObjectFactory();
    std::unique_ptr<IFisheyeDewarper> getFisheyeDewarper();
    std::unique_ptr<IDetectionFisheyeDewarper> getDetectionFisheyeDewarper(float aspectRatio);
    std::unique_ptr<ISynchronizer> getSynchronizer();
    std::unique_ptr<ISynchronizer> getDetectionSynchronizer();

private:

    bool useZeroCopyIfSupported_;
    bool isZeroCopySupported_;

};

#endif //!IMPLEMENTATION_FACTORY_H