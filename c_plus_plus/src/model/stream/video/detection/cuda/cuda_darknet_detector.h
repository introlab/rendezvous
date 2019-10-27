#ifndef CUDA_DARKNET_DETECTOR_H
#define CUDA_DARKNET_DETECTOR_H

#include "model/stream/video/detection/base_darknet_detector.h"

namespace Model
{
class CudaDarknetDetector : public BaseDarknetDetector
{
   public:
    CudaDarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);

   protected:
    image convertToDarknetImage(const ImageFloat& img) override;
    void predictImage(network* net, const image& img) override;
};

}    // namespace Model

#endif    //! CUDA_DARKNET_DETECTOR_H
