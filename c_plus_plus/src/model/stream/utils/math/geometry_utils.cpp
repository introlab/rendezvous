#include "geometry_utils.h"

#include <cmath>

#include "math_constants.h"

namespace Model
{

namespace math
{

SphericalAngleRect convertToAngleRect(const SphericalAngleBox& angleBox)
{
    float azimuthSpan = std::abs(angleBox.rightAzimuth - angleBox.leftAzimuth);
    float elevationSpan = std::abs(angleBox.topElevation - angleBox.bottomElevation);

    if (angleBox.leftAzimuth > angleBox.rightAzimuth)
    {
        azimuthSpan = (math::PI * 2.f) - azimuthSpan;
    }

    float azimuth = angleBox.leftAzimuth + azimuthSpan / 2.f;
    float elevation = angleBox.bottomElevation + elevationSpan / 2.f;

    if (azimuth > (math::PI * 2.f))
    {
        azimuth -= (math::PI * 2.f);
    }

    return SphericalAngleRect(azimuth, elevation, azimuthSpan, elevationSpan);
}

SphericalAngleBox convertToAngleBox(const SphericalAngleRect& angleRect)
{
    float leftAzimuth = angleRect.azimuth - (angleRect.azimuthSpan / 2.f);
    float rightAzimuth = angleRect.azimuth + (angleRect.azimuthSpan / 2);
    float bottomElevation = angleRect.elevation - (angleRect.elevationSpan / 2);
    float topElevation = angleRect.elevation + (angleRect.elevationSpan / 2);

    if (leftAzimuth < 0.f)
    {
        leftAzimuth += (math::PI * 2.f);
    }

    if (rightAzimuth > (math::PI * 2.f))
    {
        rightAzimuth -= (math::PI * 2.f);
    }

    return SphericalAngleBox(leftAzimuth, rightAzimuth, bottomElevation, topElevation);
}

Rectangle convertToRectangle(const BoundingBox& boundingBox)
{
    int width = std::abs(boundingBox.rightX - boundingBox.leftX);
    int height = std::abs(boundingBox.topY - boundingBox.bottomY);

    int x = boundingBox.leftX + width / 2;
    int y = boundingBox.bottomY + height / 2;

    return Rectangle(x, y, width, height);
}

BoundingBox convertToBoundingBox(const Rectangle& rectangle)
{
    int leftX = rectangle.x - (rectangle.width / 2);
    int rightX = rectangle.x + (rectangle.width / 2);
    int bottomY = rectangle.y - (rectangle.height / 2);
    int topY = rectangle.y + (rectangle.height / 2);

    return BoundingBox(leftX, rightX, bottomY, topY);
}

}
} // Model
