#include "AngleCalculations.h"

#include <cmath>
#include <stdexcept>

#include "dewarping/DewarpingHelper.h"
#include "utils/math/MathConstants.h"

namespace
{

Point<float> getSourceFromCenterDelta(const Point<float>& pixel, const Point<float>& fisheyeCenter,
                                    const DewarpingParameters& dewarpingParameters)
{
    Point<float> sourcePixel = dewarping::getSourcePixelFromDewarpedImage(pixel, dewarpingParameters);
    return Point<float>(sourcePixel.x - fisheyeCenter.x, sourcePixel.y - fisheyeCenter.y);
}

}

namespace math
{

float deg2rad(float deg)
{
    return deg * PI / 180.f;
}

float rad2deg(float rad)
{
    return rad / PI * 180.f;
}

float getElevationFromImage(const Point<float>& pixel, float fisheyeAngle, const Point<float>& fisheyeCenter, 
                            const DewarpingParameters& dewarpingParameters)
{
    if (fisheyeAngle > 2.f * PI)
    {
        throw std::invalid_argument("Fisheye angle must be in radian!");
    }

    Point<float> delta = getSourceFromCenterDelta(pixel, fisheyeCenter, dewarpingParameters);
    float distanceFromCenter = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    float distanceFromBorder = fisheyeCenter.x - distanceFromCenter;
    float ratio = (distanceFromBorder / fisheyeCenter.x);

    return ratio * (fisheyeAngle / 2.f) + (PI / 2.f - (fisheyeAngle / 2.f));
}

float getAzimuthFromImage(const Point<float>& pixel, const Point<float>& fisheyeCenter,
                          const DewarpingParameters& dewarpingParameters)
{
    Point<float> delta = getSourceFromCenterDelta(pixel, fisheyeCenter, dewarpingParameters);
    float azimuth = 0.f;

    // Based on which dial of the image the pixel is, calculate the azimuth 
    if (delta.x >= 0.f)
    {
        if (delta.y >= 0.f)
        {
            azimuth = std::atan(delta.x / delta.y);
        }
        else
        {
             azimuth = std::atan(-delta.y / delta.x) + PI / 2.f;
        }
    }
    else
    {
        if (delta.y >= 0.f)
        {
            azimuth = std::atan(delta.y / -delta.x) + 3.f * PI / 2.f;
        }
        else
        {
            azimuth = std::atan(-delta.x / -delta.y) + PI;
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

float getLinearApproximatedSphericalAnglesDistance(float srcAzimuth, float srcElevation, float dstAzimuth, float dstElevation)
{
    float azimuthDistance = getSignedAzimuthDifference(srcAzimuth, dstAzimuth);
    float elevationDistance = srcElevation - dstElevation;

    return std::sqrt(azimuthDistance * azimuthDistance + elevationDistance * elevationDistance);
}

float getApproximatedSphericalAnglesDistance(float srcElevation, float azimuthDifference, float elevationDifference)
{
    float absAzimuthDifference = getSmallestAbsAzimuthDifference(std::abs(azimuthDifference));
    
    float azimuthArcDistance = std::sin((math::PI / 2.f) - (srcElevation + elevationDifference / 2.f)) * absAzimuthDifference;
    float elevationArcDistance = std::abs(elevationDifference);

    return std::sqrt(azimuthArcDistance * azimuthArcDistance + elevationArcDistance * elevationArcDistance);
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
           std::cos(math::PI * 2.f - srcElevation) * std::cos(math::PI * 2.f - dstElevation) * std::cos(absAzimuthDifference));
}

}
