#ifndef I_IMAGE_CONVERTER_H
#define I_IMAGE_CONVERTER_H

#include "utils/images/Images.h"

class IImageConverter
{
public:

    virtual ~IImageConverter() {};
    virtual void convert(const Image& inImage, const Image& outImage) = 0;

};

#endif //!I_IMAGE_CONVERTER_H