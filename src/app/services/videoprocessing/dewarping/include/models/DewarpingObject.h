#ifndef DEWARPING_OBJECT_H
#define DEWARPING_OBJECT_H

#include <models/DewarpingParameters.h>
#include <models/ImageBuffer.h>

struct DewarpingObject
{
    DewarpingObject(int renderContextId, ImageBuffer& imageBuffer, DewarpingParameters& dewarpingParameters);
    DewarpingObject(int renderContextId, ImageBuffer& imageBuffer);

    int renderContextId;
    ImageBuffer imageBuffer;
    DewarpingParameters dewarpingParameters;
    bool isDewarping;
};

#endif //!DEWARPING_OBJECT_H