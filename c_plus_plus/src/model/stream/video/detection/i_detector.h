#ifndef I_DETECTOR_H
#define I_DETECTOR_H

#include <vector>

#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/rectangle.h"

namespace Model
{

class IDetector
{
public:

    virtual ~IDetector() = default;
    virtual std::vector<Rectangle> detectInImage(const ImageFloat& image) = 0;
    virtual Dim2<int> getInputImageDim() = 0;

};

} // Model

#endif //!I_DETECTOR_H
