#include "cuda_darknet_detector.h"

namespace Model
{
CudaDarknetDetector::CudaDarknetDetector(const std::string& configFile, const std::string& weightsFile,
                                         const std::string& metaFile, int sleepBetweenLayersForwardUs)
    : BaseDarknetDetector(configFile, weightsFile, metaFile, sleepBetweenLayersForwardUs)
{
}

image CudaDarknetDetector::convertToDarknetImage(const ImageFloat& img)
{
    image darknetImage;
    darknetImage.w = img.width;
    darknetImage.h = img.height;
    darknetImage.c = 3;
    darknetImage.data = img.deviceData;

    return darknetImage;
}

void CudaDarknetDetector::predictImage(network* net, const image& img)
{
    network_predict_letterbox_gpu_device_image(net, img);
}
}    // namespace Model