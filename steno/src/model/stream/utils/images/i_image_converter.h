#ifndef I_IMAGE_CONVERTER_H
#define I_IMAGE_CONVERTER_H

#include "model/stream/utils/images/images.h"

namespace Model
{
class IImageConverter
{
   public:
    virtual ~IImageConverter() = default;
    virtual void convert(const Image& inImage, Image& outImage) = 0;
};

}    // namespace Model

#endif    //! I_IMAGE_CONVERTER_H
