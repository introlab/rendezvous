#include "dewarping_helper.h"

#include <cmath>
#include <stdexcept>

#include "model/stream/utils/math/angle_calculations.h"
#include "model/stream/utils/math/geometry_utils.h"
#include "model/stream/utils/math/math_constants.h"

namespace Model
{
DonutSlice createDewarpingDonutSlice(DonutSlice& baseDonutSlice, float centersDistance)
{
    if (baseDonutSlice.middleAngle > 2 * math::PI)
    {
        throw std::invalid_argument("Donut slice middle angle must be between 0 and 2*PI rad!");
    }
    else if (baseDonutSlice.angleSpan > 2 * math::PI)
    {
        throw std::invalid_argument("Donut slice angle span must be between 0 and 2*PI rad!");
    }

    float xNewCenter = baseDonutSlice.xCenter - sin(baseDonutSlice.middleAngle) * centersDistance;
    float yNewCenter = baseDonutSlice.yCenter - cos(baseDonutSlice.middleAngle) * centersDistance;

    // Length of a line which start from the new center, pass by image center and end to form the side of a right -
    // angled triangle which other point is on the circle of center(xImageCenter, yImageCenter) and radius(outRadius)
    float d = cos(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius + centersDistance;

    float newInRadius = baseDonutSlice.inRadius + centersDistance;
    float newOutRadius = sqrt(pow(d, 2) + pow(sin(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius, 2));
    float newAngleSpan = acos(d / newOutRadius) * 2;

    return DonutSlice(xNewCenter, yNewCenter, newInRadius, newOutRadius, baseDonutSlice.middleAngle, newAngleSpan);
}

DewarpingParameters getDewarpingParameters(const Dim2<int>& imageSize, const DewarpingConfig& dewarpingConfig,
                                           float middleAngle)
{
    DonutSlice donutSlice(static_cast<float>(imageSize.width) / 2.f, static_cast<float>(imageSize.height) / 2.f,
                          dewarpingConfig.inRadius, dewarpingConfig.outRadius, middleAngle, dewarpingConfig.angleSpan);

    return getDewarpingParameters(donutSlice, dewarpingConfig.topDistorsionFactor,
                                  dewarpingConfig.bottomDistorsionFactor);
}

DewarpingParameters getDewarpingParameters(DonutSlice& baseDonutSlice, float topDistorsionFactor,
                                           float bottomDistorsionFactor)
{
    // Distance between center of baseDonutSlice and newDonutSlice to calculate
    float centersDistance = topDistorsionFactor * 10000;

    // Return a new donut mapping based on the one passed, which have properties to reduce the distorsion in the image
    DonutSlice newDonutSlice = createDewarpingDonutSlice(baseDonutSlice, centersDistance);

    return getDewarpingParametersFromNewDonutSlice(baseDonutSlice, newDonutSlice, centersDistance,
                                                   bottomDistorsionFactor);
}

DewarpingParameters getDewarpingParametersFromNewDonutSlice(DonutSlice& baseDonutSlice, DonutSlice& newDonutSlice,
                                                            float centersDistance, float bottomDistorsionFactor)
{
    // Radius of circle which would be in the middle of the dewarped image if no radius factor was applied to dewarping
    float centerRadius = (newDonutSlice.inRadius + newDonutSlice.outRadius) / 2;

    // Offset in x in order for the mapping to be in the right section of the source image(changes the angle of mapping)
    float xOffset = (newDonutSlice.middleAngle - newDonutSlice.angleSpan / 2) * centerRadius;

    // Difference between the outside radius of the base donut and the new one
    float outRadiusDiff = baseDonutSlice.outRadius + centersDistance - newDonutSlice.outRadius;

    // Width and Height of the dewarped image
    float dewarpHeight = newDonutSlice.outRadius - newDonutSlice.inRadius;
    float dewarpWidth = newDonutSlice.angleSpan * centerRadius;

    return DewarpingParameters(newDonutSlice.xCenter, newDonutSlice.yCenter, dewarpWidth, dewarpHeight,
                               newDonutSlice.inRadius, centerRadius, outRadiusDiff, xOffset, bottomDistorsionFactor);
}

DewarpingParameters getDewarpingParametersFromAngleBoundingBox(const SphericalAngleRect& angleRect,
                                                               const Point<float>& fisheyeCenter,
                                                               const DewarpingConfig& dewarpingConfig)
{
    SphericalAngleBox angleBox = math::convertToAngleBox(angleRect);
    DonutSlice donutSlice(fisheyeCenter.x, fisheyeCenter.y, dewarpingConfig.inRadius, dewarpingConfig.outRadius,
                          angleRect.azimuth, angleRect.azimuthSpan);
    DewarpingParameters dewarpingParameters =
        getDewarpingParameters(donutSlice, dewarpingConfig.topDistorsionFactor, dewarpingConfig.bottomDistorsionFactor);

    float maxElevation = math::getElevationFromImage(Point<float>(dewarpingParameters.dewarpWidth / 2.f, 0),
                                                     dewarpingConfig.fisheyeAngle, fisheyeCenter, dewarpingParameters);
    float minElevation = math::getElevationFromImage(
        Point<float>(dewarpingParameters.dewarpWidth / 2.f, dewarpingParameters.dewarpHeight),
        dewarpingConfig.fisheyeAngle, fisheyeCenter, dewarpingParameters);

    float deltaElevation = maxElevation - minElevation;
    float deltaElevationTop = maxElevation - angleBox.topElevation;
    float deltaElevationBottom = angleBox.bottomElevation - minElevation;

    dewarpingParameters.topOffset = (deltaElevationTop * dewarpingParameters.dewarpHeight) / deltaElevation;
    dewarpingParameters.bottomOffset = (deltaElevationBottom * dewarpingParameters.dewarpHeight) / deltaElevation;

    return dewarpingParameters;
}

Point<float> getSourcePixelFromDewarpedImage(const Point<float>& pixel, const DewarpingParameters& dewarpingParameters)
{
    float xRadiusFactor = sinf((math::PI * pixel.x) / dewarpingParameters.dewarpWidth);
    float yRadiusFactor = sinf((math::PI * pixel.y) / (dewarpingParameters.dewarpHeight * 2.f));

    float radius = pixel.y + dewarpingParameters.inRadius +
                   dewarpingParameters.outRadiusDiff * xRadiusFactor * yRadiusFactor *
                       (1 - dewarpingParameters.bottomDistorsionFactor);
    float theta = (pixel.x + dewarpingParameters.xOffset) / dewarpingParameters.centerRadius;
    float x = dewarpingParameters.xCenter + radius * sinf(theta);
    float y = dewarpingParameters.yCenter + radius * cosf(theta);

    return Point<float>(x, y);
}

SphericalAngleRect getAngleRectFromDewarpedImageRectangle(const Rectangle& rectangle,
                                                          const DewarpingParameters& dewarpingParameters,
                                                          const Dim2<int>& imageSize, const Point<float>& fisheyeCenter,
                                                          float fisheyeAngle)
{
    BoundingBox boundingBox = math::convertToBoundingBox(rectangle);

    float dewarpWidthFactor = dewarpingParameters.dewarpWidth / static_cast<float>(imageSize.width);
    float dewarpHeightFactor = dewarpingParameters.dewarpHeight / static_cast<float>(imageSize.height);

    float leftX = static_cast<float>(boundingBox.leftX) * dewarpWidthFactor;
    float rightX = static_cast<float>(boundingBox.rightX) * dewarpWidthFactor;
    float bottomY = static_cast<float>(boundingBox.bottomY) * dewarpHeightFactor;
    float topY = static_cast<float>(boundingBox.topY) * dewarpHeightFactor;

    float xMostTop, xMostBottom, yMostLeft, yMostRight;

    if (rectangle.x > imageSize.width / 2.f)
    {
        xMostTop = leftX;
        xMostBottom = rightX;
        yMostLeft = topY;
        yMostRight = bottomY;
    }
    else
    {
        xMostTop = rightX;
        xMostBottom = leftX;
        yMostLeft = topY;
        yMostRight = bottomY;
    }

    // Are the bottomY and topY inverted?
    float azimuthLeft = math::getAzimuthFromImage(Point<float>(leftX, yMostLeft), fisheyeCenter, dewarpingParameters);
    float azimuthRight =
        math::getAzimuthFromImage(Point<float>(rightX, yMostRight), fisheyeCenter, dewarpingParameters);
    float elevationTop =
        math::getElevationFromImage(Point<float>(xMostTop, bottomY), fisheyeAngle, fisheyeCenter, dewarpingParameters);
    float elevationBottom =
        math::getElevationFromImage(Point<float>(xMostBottom, topY), fisheyeAngle, fisheyeCenter, dewarpingParameters);

    return math::convertToAngleRect(SphericalAngleBox(azimuthLeft, azimuthRight, elevationBottom, elevationTop));
}

Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index)
{
    float x = index % dst.width;
    float y = index / dst.width;

    float textureCoordsX = x / dst.width;
    float textureCoordsY = y / dst.height;

    float heightFactor = (1 - ((params.bottomOffset + params.topOffset) / params.dewarpHeight)) * textureCoordsY +
                         params.topOffset / params.dewarpHeight;
    float factor = params.outRadiusDiff * (1 - params.bottomDistorsionFactor) * std::sin(math::PI * textureCoordsX) *
                   std::sin((math::PI * heightFactor) / 2.0);
    float radius = textureCoordsY * params.dewarpHeight + params.inRadius + factor +
                   (1 - textureCoordsY) * params.topOffset - textureCoordsY * params.bottomOffset;
    float theta = ((textureCoordsX * params.dewarpWidth) + params.xOffset) / params.centerRadius;

    Point<float> srcPixelPosition;
    srcPixelPosition.x = params.xCenter + radius * std::sin(theta);
    srcPixelPosition.y = params.yCenter + radius * std::cos(theta);

    return srcPixelPosition;
}

LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim2<int>& dim)
{
    int xRoundDown = int(pixel.x);
    int yRoundDown = int(pixel.y);
    float xRatio = pixel.x - xRoundDown;
    float yRatio = pixel.y - yRoundDown;
    float xOpposite = 1 - xRatio;
    float yOpposite = 1 - yRatio;

    LinearPixelFilter linearPixelFilter;

    linearPixelFilter.pc1.index = (xRoundDown + (yRoundDown * dim.width)) * 3;
    linearPixelFilter.pc2.index = linearPixelFilter.pc1.index + 3;
    linearPixelFilter.pc3.index = linearPixelFilter.pc1.index + dim.width * 3;
    linearPixelFilter.pc4.index = linearPixelFilter.pc2.index + dim.width * 3;

    linearPixelFilter.pc1.ratio = xOpposite * yOpposite;
    linearPixelFilter.pc2.ratio = xRatio * yOpposite;
    linearPixelFilter.pc3.ratio = xOpposite * yRatio;
    linearPixelFilter.pc4.ratio = xRatio * yRatio;

    return linearPixelFilter;
}

int calculateSourcePixelIndex(const Point<float>& pixel, const Dim2<int>& dim)
{
    return (int(pixel.x) + int(pixel.y) * dim.width) * 3;
}

}    // namespace Model
