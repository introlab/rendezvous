#ifndef DARKNET_DETECTOR_H
#define DARKNET_DETECTOR_H

#include "model/stream/video/detection/base_darknet_detector.h"

namespace Model
{

class DarknetDetector : public BaseDarknetDetector
{
public:

    DarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);

protected:

    image convertToDarknetImage(const ImageFloat& img) override;
    void predictImage(network *net, const image& img) override;

};

} // Model

#endif //!DARKNET_DETECTOR_H
