#ifndef I_DETECTOR_H
#define I_DETECTOR_H

#include <vector>

#include "utils/images/Images.h"
#include "utils/models/Rectangle.h"

class IDetector
{
public:

    virtual ~IDetector() = default;
    virtual std::vector<Rectangle> detectInImage(const ImageFloat& image) = 0;
    virtual Dim2<int> getInputImageDim() = 0;

};

#endif //!I_DETECTOR_H