#ifndef ANGLE_CALCULATIONS_H
#define ANGLE_CALCULATIONS_H

#include "dewarping/models/DewarpingParameters.h"
#include "utils/models/Point.h"

namespace math
{

float deg2rad(float deg);
float rad2deg(float rad);

float getElevationFromImage(const Point<float>& pixel, float fisheyeAngle, const Point<float>& fisheyeCenter,
                            const DewarpingParameters& dewarpingParameters);
float getAzimuthFromImage(const Point<float>& pixel, const Point<float>& fisheyeCenter, 
                          const DewarpingParameters& dewarpingParameters);

float getSmallestAbsAzimuthDifference(float srcAzimuth, float dstAzimuth);
float getSignedAzimuthDifference(float srcAzimuth, float dstAzimuth);
float getLinearApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation);
float getApproximatedSphericalAnglesDistance(float srcElevation, float azimuthDifference, float elevationDifference);
float getApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation);
float getSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation);

}



#endif //!ANGLE_CALCULATIONS_H