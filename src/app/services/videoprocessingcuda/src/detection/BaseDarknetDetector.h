#ifndef BASE_DARKNET_DETECTOR_H
#define BASE_DARKNET_DETECTOR_H

#include <string>

#ifndef NO_CUDA
#define GPU
#endif

#include "darknet.h"

#include "IDetector.h"

class BaseDarknetDetector : public IDetector
{
public:

    BaseDarknetDetector(const std::string& configFile, const std::string& weightsFile, const std::string& metaFile);
    virtual ~BaseDarknetDetector();

    std::vector<Rectangle> detectInImage(const ImageFloat& img) override;
    Dim2<int> getInputImageDim() override;
    
protected:

    virtual image convertToDarknetImage(const ImageFloat& img) = 0;
    virtual void predictImage(network *net, const image& img) = 0;

private:

    std::vector<Rectangle> detect(const image& img, float thresh, float hier_thresh, float nms);

    network* network_;
    metadata metadata_;
};

#endif //!BASE_DARKNET_DETECTOR_H