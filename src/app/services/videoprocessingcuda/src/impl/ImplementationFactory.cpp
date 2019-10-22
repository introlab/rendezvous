#include "ImplementationFactory.h"

#include <iostream>

#ifdef NO_CUDA
#include "detection/DarknetDetector.h"
#include "dewarping/CpuFisheyeDewarper.h"
#include "dewarping/CpuDarknetFisheyeDewarper.h"
#include "utils/alloc/HeapObjectFactory.h"
#include "utils/threads/sync/NopSynchronizer.h"
#else
#include "cuda_runtime.h"
#include "detection/cuda/CudaDarknetDetector.h"
#include "dewarping/cuda/CudaFisheyeDewarper.h"
#include "dewarping/cuda/CudaDarknetFisheyeDewarper.h"
#include "utils/alloc/cuda/DeviceCudaObjectFactory.h"
#include "utils/alloc/cuda/ManagedMemoryCudaObjectFactory.h"
#include "utils/alloc/cuda/ZeroCopyCudaObjectFactory.h"
#include "utils/threads/sync/CudaSynchronizer.h"

namespace
{
    cudaStream_t stream;
    cudaStream_t detectionStream;
}

#endif

ImplementationFactory::ImplementationFactory(bool useZeroCopyIfSupported)
    : useZeroCopyIfSupported_(useZeroCopyIfSupported)
    , isZeroCopySupported_(false)
{
    std::string message;
#ifdef NO_CUDA
    message = "Application was compiled without CUDA, calculations will be executed on the CPU.";
#else
    const int deviceNumber = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNumber);
    isZeroCopySupported_ = deviceProp.integrated;
    message = isZeroCopySupported_ ? "Graphic card supports zero copy" : "Graphic card doesn't support zero copy";

    if (!useZeroCopyIfSupported_ || !isZeroCopySupported_)
    {
        cudaStreamCreate(&stream);
        cudaStreamCreate(&detectionStream);
    }

#endif
    std::cout << message << std::endl;
}

std::unique_ptr<IDetector> ImplementationFactory::getDetector(const std::string& configFile, const std::string& weightsFile, 
                                                              const std::string& metaFile)
{
    std::unique_ptr<IDetector> detector = nullptr;

#ifdef NO_CUDA
    detector = std::make_unique<DarknetDetector>(configFile, weightsFile, metaFile);
#else
    detector = std::make_unique<CudaDarknetDetector>(configFile, weightsFile, metaFile);
#endif

    return detector;
}

std::unique_ptr<IObjectFactory> ImplementationFactory::getObjectFactory()
{
    std::unique_ptr<IObjectFactory> objectFactory = nullptr;

#ifdef NO_CUDA
    objectFactory = std::make_unique<HeapObjectFactory>();
#else
    if (useZeroCopyIfSupported_ && isZeroCopySupported_)
    {
        objectFactory = std::make_unique<ZeroCopyCudaObjectFactory>();
    }
    else
    {
        objectFactory = std::make_unique<ManagedMemoryCudaObjectFactory>(stream);
    }     
#endif

    return objectFactory;
}

std::unique_ptr<IObjectFactory> ImplementationFactory::getDetectionObjectFactory()
{
     std::unique_ptr<IObjectFactory> objectFactory = nullptr;

#ifdef NO_CUDA
    objectFactory = std::make_unique<HeapObjectFactory>();
#else
    if (useZeroCopyIfSupported_ && isZeroCopySupported_)
    {
        objectFactory = std::make_unique<ZeroCopyCudaObjectFactory>();
    }
    else
    {
        objectFactory = std::make_unique<DeviceCudaObjectFactory>();
    }     
#endif

    return objectFactory;
}

std::unique_ptr<IFisheyeDewarper> ImplementationFactory::getFisheyeDewarper()
{
    std::unique_ptr<IFisheyeDewarper> fisheyeDewarper = nullptr;

#ifdef NO_CUDA
    fisheyeDewarper = std::make_unique<CpuFisheyeDewarper>();
#else
    fisheyeDewarper = std::make_unique<CudaFisheyeDewarper>(stream);
#endif

    return fisheyeDewarper;
}

std::unique_ptr<IDetectionFisheyeDewarper> ImplementationFactory::getDetectionFisheyeDewarper(float aspectRatio)
{
    std::unique_ptr<IDetectionFisheyeDewarper> normalizedFisheyeDewarper = nullptr;

#ifdef NO_CUDA
    normalizedFisheyeDewarper = std::make_unique<CpuDarknetFisheyeDewarper>(aspectRatio);
#else
    normalizedFisheyeDewarper = std::make_unique<CudaDarknetFisheyeDewarper>(detectionStream, aspectRatio);
#endif

    return normalizedFisheyeDewarper;
}

std::unique_ptr<ISynchronizer> ImplementationFactory::getSynchronizer()
{
    std::unique_ptr<ISynchronizer> synchronizer = nullptr;

#ifdef NO_CUDA
    synchronizer = std::make_unique<NopSynchronizer>();
#else
    synchronizer = std::make_unique<CudaSynchronizer>(stream);
#endif

    return synchronizer;
}

std::unique_ptr<ISynchronizer> ImplementationFactory::getDetectionSynchronizer()
{
    std::unique_ptr<ISynchronizer> synchronizer = nullptr;

#ifdef NO_CUDA
    synchronizer = std::make_unique<NopSynchronizer>();
#else
    synchronizer = std::make_unique<CudaSynchronizer>(detectionStream);
#endif

    return synchronizer;
}