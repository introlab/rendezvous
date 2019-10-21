#ifndef DARKNET_DETECTOR_H
#define DARKNET_DETECTOR_H

#include "detection/BaseDarknetDetector.h"

class DarknetDetector : public BaseDarknetDetector
{
public:

    DarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);

protected:

    image convertToDarknetImage(const ImageFloat& img) override;
    void predictImage(network *net, const image& img) override;

};

#endif //!DARKNET_DETECTOR_H