#ifndef I_VIDEO_OUTPUT_H
#define I_VIDEO_OUTPUT_H

#include "model/stream/utils/images/images.h"

namespace Model
{

class IVideoOutput
{
public:

    virtual ~IVideoOutput() = default;
    virtual void writeImage(const Image& image) = 0;

};

} // Model

#endif // !I_VIDEO_OUTPUT_H

