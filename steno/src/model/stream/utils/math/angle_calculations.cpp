#include "angle_calculations.h"

#include <cmath>
#include <stdexcept>

#include "model/stream/utils/math/math_constants.h"
#include "model/stream/utils/math/helpers.h"

namespace Model
{
namespace math
{
float deg2rad(float deg)
{
    return deg * math::PI / 180.f;
}

float rad2deg(float rad)
{
    return rad / math::PI * 180.f;
}

float getAngleAroundCircle(float angle)
{
    if (angle < 0.f)
    {
        angle += math::PI * 2.f;
    }
    else if (angle > math::PI * 2.f)
    {
        angle = std::fmod(angle, math::PI * 2.f);
    }

    return angle;
}

float getPositiveAngle(float angle)
{
    if (angle < 0.f)
    {
        angle += math::PI * 2.f;
    }

    return angle;
}

float getAzimuthFromPosition(float x, float y)
{
    float tanRes = std::atan2(y, x);
    return getPositiveAngle(tanRes);
}

float getElevationFromPosition(float x, float y, float z)
{
    float xyHypotenuse = euclideanDistance(x, y);
    float tanRes = std::atan2(z, xyHypotenuse);
    return getPositiveAngle(tanRes);
}

float getElevationFromDistanceToFisheyeCenter(float distanceToFisheyeCenter, float fisheyeRadius, float fisheyeAngle)
{
    float distanceFromFisheyeEdge = fisheyeRadius - distanceToFisheyeCenter;
    float distanceRatio = (distanceFromFisheyeEdge / fisheyeRadius);
    float fisheyeElevationSpan = fisheyeAngle / 2.f;
    float elevation = distanceRatio * fisheyeElevationSpan + (math::PI / 2.f - fisheyeElevationSpan);

    return elevation;
}

float getDistanceToFisheyeCenterFromElevation(float elevation, float fisheyeRadius, float fisheyeAngle)
{
    float fisheyeMaxElevationSpan = fisheyeAngle / 2.f;
    float ratio = (elevation - (math::PI / 2.f - fisheyeMaxElevationSpan)) / fisheyeMaxElevationSpan;
    float distanceFromBorder = ratio * fisheyeRadius;
    float distanceToFisheyeCenter = fisheyeRadius - distanceFromBorder;

    return distanceToFisheyeCenter;
}

float getAzimuthFromDistanceToFisheyeCenter(const Point<float>& distanceToFisheyeCenter)
{
    float azimuth = 0.f;

    // Based on which dial of the image the pixel is, calculate the azimuth
    if (distanceToFisheyeCenter.x >= 0.f)
    {
        if (distanceToFisheyeCenter.y >= 0.f)
        {
            azimuth = std::atan(distanceToFisheyeCenter.x / distanceToFisheyeCenter.y);
        }
        else
        {
            azimuth = std::atan(-distanceToFisheyeCenter.y / distanceToFisheyeCenter.x) + math::PI / 2.f;
        }
    }
    else
    {
        if (distanceToFisheyeCenter.y >= 0.f)
        {
            azimuth = std::atan(distanceToFisheyeCenter.y / -distanceToFisheyeCenter.x) + 3.f * math::PI / 2.f;
        }
        else
        {
            azimuth = std::atan(-distanceToFisheyeCenter.x / -distanceToFisheyeCenter.y) + math::PI;
        }
    }

    return azimuth;
}

float getSmallestAbsAzimuthDifference(float absDifference)
{
    // Shortest distance around a circle is always smaller or equal to 180 degrees
    if (absDifference > math::PI)
    {
        absDifference = math::PI * 2.f - absDifference;
    }

    return absDifference;
}

float getSignedAzimuthDifference(float srcAzimuth, float dstAzimuth)
{
    float difference = dstAzimuth - srcAzimuth;
    float absDifference = getSmallestAbsAzimuthDifference(std::abs(difference));

    // Determine if shortest distance is clockwise (positive) or anti-clockwise (negative)
    float angleCheck = difference < 0 ? -math::PI : math::PI;
    return difference < angleCheck ? absDifference : -absDifference;
}

float getLinearApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth,
                                                   float dstElevation)
{
    float azimuthDistance = getSignedAzimuthDifference(srcAzimuth, dstAzimuth);
    float elevationDistance = srcElevation - dstElevation;

    return euclideanDistance(azimuthDistance, elevationDistance);
}

float getApproximatedSphericalAnglesDistance(float srcElevation, float azimuthDifference, float elevationDifference)
{
    float absAzimuthDifference = getSmallestAbsAzimuthDifference(std::abs(azimuthDifference));

    float azimuthArcDistance =
        std::sin((math::PI / 2.f) - (srcElevation + elevationDifference / 2.f)) * absAzimuthDifference;
    float elevationArcDistance = std::abs(elevationDifference);

    return euclideanDistance(azimuthArcDistance, elevationArcDistance);
}

float getApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation)
{
    float azimuthDifference = dstAzimuth - srcAzimuth;
    float elevationDifference = dstElevation - srcElevation;

    return getApproximatedSphericalAnglesDistance(srcElevation, azimuthDifference, elevationDifference);
}

float getSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation)
{
    float absAzimuthDifference = getSmallestAbsAzimuthDifference(std::abs(dstAzimuth - srcAzimuth));

    return std::acos(std::sin(math::PI * 2.f - srcElevation) * std::sin(math::PI * 2.f - dstElevation) +
                     std::cos(math::PI * 2.f - srcElevation) * std::cos(math::PI * 2.f - dstElevation) *
                     std::cos(absAzimuthDifference));
}
}    // namespace math
}    // namespace Model
