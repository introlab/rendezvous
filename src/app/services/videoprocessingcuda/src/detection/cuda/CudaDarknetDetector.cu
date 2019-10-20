#include "CudaDarknetDetector.h"

CudaDarknetDetector::CudaDarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile)
    : BaseDarknetDetector(configFile, weightsFile, metaFile)
{
}

image CudaDarknetDetector::convertToDarknetImage(const ImageFloat& img)
{
    image darknetImage;
    darknetImage.w = img.width;
    darknetImage.h = img.height;
    darknetImage.c = img.channels;
    darknetImage.data = img.deviceData;

    return darknetImage;
}

void CudaDarknetDetector::predictImage(network *net, const image& img)
{
    network_predict_letterbox_gpu_device_image(net, img);
}