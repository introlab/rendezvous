#ifndef ANGLE_CALCULATIONS_H
#define ANGLE_CALCULATIONS_H

#include "model/stream/utils/models/point.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"

namespace Model
{
namespace math
{
float deg2rad(float deg);
float rad2deg(float rad);

float getAngleAroundCircle(float angle);
float getPositiveAngle(float angle);

float getAzimuthFromPosition(float x, float y);
float getElevationFromPosition(float x, float y, float z);

float getElevationFromDistanceToFisheyeCenter(float distanceToFisheyeCenter, float fisheyeRadius, float fisheyeAngle);
float getDistanceToFisheyeCenterFromElevation(float elevation, float fisheyeRadius, float fisheyeAngle);
float getAzimuthFromDistanceToFisheyeCenter(const Point<float>& distanceToFisheyeCenter);

float getSmallestAbsAzimuthDifference(float absDifference);
float getSignedAzimuthDifference(float srcAzimuth, float dstAzimuth);
float getLinearApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth,
                                                   float dstElevation);
float getApproximatedSphericalAnglesDistance(float srcElevation, float azimuthDifference, float elevationDifference);
float getApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth,
                                             float dstElevation);
float getSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation);

}    // namespace math

}    // namespace Model

#endif    //! ANGLE_CALCULATIONS_H
