#include <models/DewarpingObject.h>

DewarpingObject::DewarpingObject(int renderContextId, ImageBuffer& imageBuffer, DewarpingParameters& dewarpingParameters)
    : renderContextId(renderContextId),
    imageBuffer(imageBuffer),
    dewarpingParameters(dewarpingParameters),
    isDewarping(true)
{
}

DewarpingObject::DewarpingObject(int renderContextId, ImageBuffer& imageBuffer)
    : renderContextId(renderContextId),
    imageBuffer(imageBuffer),
    isDewarping(false)
{
}