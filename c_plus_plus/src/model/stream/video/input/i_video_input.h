#ifndef I_VIDEO_INPUT_H
#define I_VIDEO_INPUT_H

#include "model/stream/utils/images/images.h"

namespace Model
{
class IVideoInput
{
   public:
    virtual ~IVideoInput() = default;
    virtual const Image& readImage() = 0;
};

}    // namespace Model

#endif    // !I_VIDEO_INPUT_H
