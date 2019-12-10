#include "implementation_factory.h"

#include <iostream>

#ifdef NO_CUDA
#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_converter.h"
#include "model/stream/utils/threads/sync/nop_synchronizer.h"
#include "model/stream/video/detection/darknet_detector.h"
#include "model/stream/video/dewarping/cpu_darknet_fisheye_dewarper.h"
#include "model/stream/video/dewarping/cpu_fisheye_dewarper.h"
#include "model/stream/video/input/camera_reader.h"
#include "model/stream/video/input/vc_camera_reader.h"
#include "model/stream/video/input/image_file_reader.h"
#else
#include "cuda_runtime.h"
#include "model/stream/utils/alloc/cuda/device_cuda_object_factory.h"
#include "model/stream/utils/alloc/cuda/managed_memory_cuda_object_factory.h"
#include "model/stream/utils/alloc/cuda/zero_copy_cuda_object_factory.h"
#include "model/stream/utils/images/cuda/cuda_image_converter.h"
#include "model/stream/utils/threads/sync/cuda_synchronizer.h"
#include "model/stream/video/detection/cuda/cuda_darknet_detector.h"
#include "model/stream/video/dewarping/cuda/cuda_darknet_fisheye_dewarper.h"
#include "model/stream/video/dewarping/cuda/cuda_fisheye_dewarper.h"
#include "model/stream/video/input/cuda/cuda_camera_reader.h"
#include "model/stream/video/input/cuda/vc_cuda_camera_reader.h"
#include "model/stream/video/input/cuda/cuda_image_file_reader.h"

namespace
{
cudaStream_t stream;
cudaStream_t detectionStream;
}    // namespace

#endif

namespace Model
{
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

    cudaStreamCreate(&stream);
    cudaStreamCreate(&detectionStream);
#endif
    std::cout << message << std::endl;
}

ImplementationFactory::~ImplementationFactory()
{
#ifndef NO_CUDA
    cudaStreamDestroy(stream);
    cudaStreamDestroy(detectionStream);
#endif
}

std::unique_ptr<IDetector> ImplementationFactory::getDetector(const std::string& configFile,
                                                              const std::string& weightsFile,
                                                              const std::string& metaFile,
                                                              int sleepBetweenLayersForwardUs)
{
    std::unique_ptr<IDetector> detector = nullptr;

#ifdef NO_CUDA
    detector = std::make_unique<DarknetDetector>(configFile, weightsFile, metaFile, sleepBetweenLayersForwardUs);
#else
    detector = std::make_unique<CudaDarknetDetector>(configFile, weightsFile, metaFile, sleepBetweenLayersForwardUs);
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

std::unique_ptr<IImageConverter> ImplementationFactory::getImageConverter()
{
    std::unique_ptr<IImageConverter> imageConverter = nullptr;

#ifdef NO_CUDA
    imageConverter = std::make_unique<ImageConverter>();
#else
    imageConverter = std::make_unique<CudaImageConverter>(stream);
#endif

    return imageConverter;
}

std::unique_ptr<IVideoInput> ImplementationFactory::getImageFileReader(const std::string& imageFilePath,
                                                                       ImageFormat format)
{
    std::unique_ptr<IVideoInput> fileImageReader = nullptr;

#ifdef NO_CUDA
    fileImageReader = std::make_unique<ImageFileReader>(imageFilePath, format);
#else
    fileImageReader = std::make_unique<CudaImageFileReader>(imageFilePath, format);
#endif

    return fileImageReader;
}

std::unique_ptr<IVideoInput> ImplementationFactory::getCameraReader(std::shared_ptr<VideoConfig> videoConfig)
{
    std::unique_ptr<IVideoInput> cameraReader = nullptr;

#ifdef NO_CUDA
    cameraReader = std::make_unique<CameraReader>(videoConfig, 2);
#else
    cameraReader = std::make_unique<CudaCameraReader>(videoConfig);
#endif

    return cameraReader;
}

std::unique_ptr<IVideoInput> ImplementationFactory::getVcCameraReader(std::shared_ptr<VideoConfig> videoConfig)
{
    std::unique_ptr<IVideoInput> cameraReader = nullptr;

#ifdef NO_CUDA
    cameraReader = std::make_unique<VcCameraReader>(videoConfig, 2);
#else
    cameraReader = std::make_unique<VcCudaCameraReader>(videoConfig);
#endif

    return cameraReader;
}
}    // namespace Model
