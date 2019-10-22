#ifndef I_VIDEO_OUTPUT_H
#define I_VIDEO_OUTPUT_H

#include "utils/images/Images.h"

class IVideoOutput
{
public:

    virtual ~IVideoOutput() {};
    virtual void writeImage(const Image& image) = 0;

};

#endif // !I_VIDEO_OUTPUT_H
