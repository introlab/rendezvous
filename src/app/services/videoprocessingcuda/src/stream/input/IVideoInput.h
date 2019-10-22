#ifndef I_VIDEO_INPUT_H
#define I_VIDEO_INPUT_H

#include "utils/images/Images.h"

class IVideoInput
{
public:

    virtual ~IVideoInput() = default;
    virtual const Image& readImage() = 0;

};

#endif // !I_VIDEO_INPUT_H
