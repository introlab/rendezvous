#include "darknet_detector.h"

namespace Model
{
DarknetDetector::DarknetDetector(const std::string& configFile, const std::string& weightsFile,
                                 const std::string& metaFile, int sleepBetweenLayersForwardUs)
    : BaseDarknetDetector(configFile, weightsFile, metaFile, sleepBetweenLayersForwardUs)
{
}

image DarknetDetector::convertToDarknetImage(const ImageFloat& img)
{
    image darknetImage;
    darknetImage.w = img.width;
    darknetImage.h = img.height;
    darknetImage.c = 3;
    darknetImage.data = img.hostData;

    return darknetImage;
}

void DarknetDetector::predictImage(network* net, const image& img)
{
    network_predict_letterbox_image(net, img);
}
}    // namespace Model
