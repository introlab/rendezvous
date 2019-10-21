#ifndef I_IMAGE_CONSUMER_H
#define I_IMAGE_CONSUMER_H

#include "utils/images/Image.h"

class IImageConsumer
{
public:

    virtual ~IImageConsumer() {};
    virtual void consumeImage(const Image& image) = 0;

};

#endif // !I_IMAGE_CONSUMER_H
