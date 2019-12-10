#ifndef I_VIDEO_INPUT_H
#define I_VIDEO_INPUT_H

#include "model/stream/utils/images/images.h"

namespace Model
{
class IVideoInput
{
   public:
    virtual ~IVideoInput() = default;
    virtual void open() = 0;
    virtual void close() = 0;
    virtual bool readImage(Image& image) = 0;
};

}    // namespace Model

#endif    // !I_VIDEO_INPUT_H
