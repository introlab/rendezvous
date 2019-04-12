#include <models/DewarpingObject.h>

DewarpingObject::DewarpingObject(int fisheyeContextId, int renderContextId, 
    ImageBuffer& imageBuffer, DewarpingParameters& dewarpingParameters)
    : fisheyeContextId(fisheyeContextId),
    renderContextId(renderContextId),
    imageBuffer(imageBuffer),
    dewarpingParameters(dewarpingParameters),
    isDewarping(true)
{
}

DewarpingObject::DewarpingObject(int fisheyeContextId, int renderContextId, ImageBuffer& imageBuffer)
    : fisheyeContextId(fisheyeContextId),
    renderContextId(renderContextId),
    imageBuffer(imageBuffer),
    isDewarping(false)
{
}