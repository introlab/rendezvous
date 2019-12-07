#include "dewarping_helper.h"

#include <cmath>
#include <stdexcept>

#include "model/stream/utils/math/angle_calculations.h"
#include "model/stream/utils/math/geometry_utils.h"
#include "model/stream/utils/math/math_constants.h"
#include "model/stream/utils/math/helpers.h"

namespace Model
{

float OVERLAP_THRESHOLD = 0.8f;

DonutSlice createDewarpingDonutSlice(const DonutSlice& baseDonutSlice, float centersDistance)
{
    if (baseDonutSlice.middleAngle > 2 * math::PI)
    {
        throw std::invalid_argument("Donut slice middle angle must be between 0 and 2*PI rad!");
    }
    else if (baseDonutSlice.angleSpan > 2 * math::PI)
    {
        throw std::invalid_argument("Donut slice angle span must be between 0 and 2*PI rad!");
    }

    float xNewCenter = baseDonutSlice.xCenter - std::sin(baseDonutSlice.middleAngle) * centersDistance;
    float yNewCenter = baseDonutSlice.yCenter - std::cos(baseDonutSlice.middleAngle) * centersDistance;

    // Length of a line which start from the new center, pass by image center and end to form the side of a right -
    // angled triangle which other point is on the circle of center(xImageCenter, yImageCenter) and radius(outRadius)
    float d = std::cos(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius + centersDistance;

    float newInRadius = baseDonutSlice.inRadius + centersDistance;
    float newOutRadius = std::sqrt(std::pow(d, 2) + std::pow(sin(baseDonutSlice.angleSpan / 2) * baseDonutSlice.outRadius, 2));
    float newAngleSpan = std::acos(d / newOutRadius) * 2;

    return DonutSlice(xNewCenter, yNewCenter, newInRadius, newOutRadius, baseDonutSlice.middleAngle, newAngleSpan);
}

DewarpingParameters getDewarpingParameters(DonutSlice& baseDonutSlice, float topDistorsionFactor,
                                           float bottomDistorsionFactor)
{
    // Distance between center of baseDonutSlice and newDonutSlice to calculate
    float centersDistance = baseDonutSlice.inRadius * topDistorsionFactor;

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

DewarpingParameters getDewarpingParametersFromSphericalAngleRect(const SphericalAngleRect& angleRect, 
                                                                 const DewarpingConfig& dewarpingConfig, 
                                                                 const Point<float>& fisheyeCenter)
{
    SphericalAngleBox angleBox = math::convertToAngleBox(angleRect);
    float topDistanceFromCenter = math::getDistanceToFisheyeCenterFromElevation(angleBox.topElevation, fisheyeCenter.x, dewarpingConfig.fisheyeAngle);
    float bottomDistanceFromCenter = math::getDistanceToFisheyeCenterFromElevation(angleBox.bottomElevation, fisheyeCenter.x, dewarpingConfig.fisheyeAngle);

    DonutSlice baseDonutSlice(fisheyeCenter.x, fisheyeCenter.y, topDistanceFromCenter, bottomDistanceFromCenter, angleRect.azimuth, angleRect.azimuthSpan);
    DewarpingParameters dewarpingParameters = getDewarpingParameters(baseDonutSlice, dewarpingConfig.topDistorsionFactor, 
                                                                     dewarpingConfig.bottomDistorsionFactor);

    return dewarpingParameters;
}

Point<float> getSourcePixelFromDewarpedImageNormalizedPixel(const Point<float>& normalizedPixel, const DewarpingParameters& dewarpingParameters)
{
    float xRadiusFactor = std::sin(math::PI * normalizedPixel.x);
    float yRadiusFactor = std::sin((math::PI * normalizedPixel.y) / 2.f);

    float dewarpDimensionX = normalizedPixel.x * dewarpingParameters.dewarpWidth;
    float dewarpDimensionY = normalizedPixel.y * dewarpingParameters.dewarpHeight;

    float radius = dewarpDimensionY + dewarpingParameters.inRadius +
                   dewarpingParameters.outRadiusDiff * xRadiusFactor * yRadiusFactor *
                   dewarpingParameters.bottomDistorsionFactor;
    float theta = (dewarpDimensionX + dewarpingParameters.xOffset) / dewarpingParameters.centerRadius;

    Point<float> sourcePixel;
    sourcePixel.x = dewarpingParameters.xCenter + radius * std::sin(theta);
    sourcePixel.y = dewarpingParameters.yCenter + radius * std::cos(theta);

    return sourcePixel;
}

int getSourcePixelIndex(const Point<float>& pixel, const Dim2<int>& dim)
{
    return (int(pixel.x) + int(pixel.y) * dim.width);
}

Point<float> getNormalizedPixelFromIndex(int index, const Dim2<int>& dim)
{
    Point<float> normalizedPoint;
    normalizedPoint.x = float(index % dim.width) / dim.width;
    normalizedPoint.y = float(index / dim.width) / dim.height;

    return normalizedPoint;
}

float getPercentageOverlap(float azimuthLeft, float azimuthRight, float azimuthLeftEdge, float azimuthRightEdge)
{
    float azimuthRightNew = math::getPositiveAngle(azimuthRight - azimuthLeft);
    float azimuthLeftEdgeNew = math::getPositiveAngle(azimuthLeftEdge - azimuthLeft);
    float azimuthRightEdgeNew = math::getPositiveAngle(azimuthRightEdge - azimuthLeft);
    float azimuthOverlap = 0.f;
    float percentageOverlap = 0.f;

    bool leftEdgeBetweenAzimuthLeftRight = azimuthLeftEdgeNew < azimuthRightNew;
    bool rightEdgeBetweenAzimuthLeftRight = azimuthRightEdgeNew < azimuthRightNew;

    if (leftEdgeBetweenAzimuthLeftRight && rightEdgeBetweenAzimuthLeftRight)
    {
        if (azimuthLeftEdgeNew < azimuthRightEdgeNew)
        {
            azimuthOverlap = azimuthRightEdgeNew - azimuthLeftEdgeNew;
        }
        else
        {
            azimuthOverlap = azimuthRightNew - (azimuthLeftEdgeNew - azimuthRightEdgeNew);
        }
    }
    else if (leftEdgeBetweenAzimuthLeftRight)
    {
        azimuthOverlap = azimuthRightNew - azimuthLeftEdgeNew;
    }
    else if (rightEdgeBetweenAzimuthLeftRight)
    {
        azimuthOverlap = azimuthRightEdgeNew;
    }
    else if (azimuthRightEdgeNew < azimuthLeftEdgeNew)
    {
        azimuthOverlap = azimuthRightNew;
    }

    if (azimuthOverlap > 0.f)
    {
        float azimuthDifference = azimuthRight - azimuthLeft;
        float absAzimuthDifference = math::getSmallestAbsAzimuthDifference(std::abs(azimuthDifference));

        percentageOverlap = azimuthOverlap / absAzimuthDifference;
    }

    return percentageOverlap;
}

bool isInOverlappingZone(const SphericalAngleRect& sphericalAngleRect, float angleSpan, float middleAngle)
{
    float azimuthLeft = math::getPositiveAngle(sphericalAngleRect.azimuth - sphericalAngleRect.azimuthSpan / 2.f);
    float azimuthRight = math::getPositiveAngle(sphericalAngleRect.azimuth + sphericalAngleRect.azimuthSpan / 2.f);
    float azimuthLeftEdge = math::getPositiveAngle(middleAngle - angleSpan / 2.f);
    float azimuthRightEdge = math::getPositiveAngle(middleAngle + angleSpan / 2.f);
    float percentageOverlap = getPercentageOverlap(azimuthLeft, azimuthRight, azimuthLeftEdge, azimuthRightEdge);

    return percentageOverlap < OVERLAP_THRESHOLD;
}

SphericalAngleRect getAngleRectFromDewarpedImageRectangle(const Rectangle& rectangle,
                                                          const DewarpingParameters& dewarpingParameters,
                                                          const Dim2<int>& imageSize, const Point<float>& fisheyeCenter,
                                                          float fisheyeAngle)
{
    BoundingBox boundingBox = math::convertToBoundingBox(rectangle);

    // Dewarped image size is not necessarily the optimal dewarp size, but all calculations need to be executed as if it was
    float dewarpWidthFactor = dewarpingParameters.dewarpWidth / static_cast<float>(imageSize.width);
    float dewarpHeightFactor = dewarpingParameters.dewarpHeight / static_cast<float>(imageSize.height);

    float leftX = static_cast<float>(boundingBox.leftX) * dewarpWidthFactor;
    float rightX = static_cast<float>(boundingBox.rightX) * dewarpWidthFactor;
    float bottomY = static_cast<float>(boundingBox.bottomY) * dewarpHeightFactor;
    float topY = static_cast<float>(boundingBox.topY) * dewarpHeightFactor;

    float middleX = static_cast<float>(rectangle.x) * dewarpWidthFactor;
    float middleY = static_cast<float>(rectangle.y) * dewarpHeightFactor;

    float azimuthLeft, azimuthRight, elevationTop, elevationBottom;
    float azimuth = getAzimuthFromDewarpedImagePixel(Point<float>(middleX, middleY), fisheyeCenter, dewarpingParameters);
    float elevation = getElevationFromDewarpedImagePixel(Point<float>(middleX, middleY), fisheyeAngle, fisheyeCenter, dewarpingParameters);

    // We take the largest angle spans possible to include the whole box (Required because dewarping is non-linear)
    if (rectangle.x > imageSize.width / 2.f)
    {
        elevationBottom = getElevationFromDewarpedImagePixel(Point<float>(rightX, bottomY), fisheyeAngle, fisheyeCenter, dewarpingParameters);
        elevationTop = getElevationFromDewarpedImagePixel(Point<float>(leftX, topY), fisheyeAngle, fisheyeCenter, dewarpingParameters);
        azimuthLeft = getAzimuthFromDewarpedImagePixel(Point<float>(leftX, bottomY), fisheyeCenter, dewarpingParameters);
        azimuthRight = getAzimuthFromDewarpedImagePixel(Point<float>(rightX, topY), fisheyeCenter, dewarpingParameters);
    }
    else
    {
        elevationBottom = getElevationFromDewarpedImagePixel(Point<float>(leftX, bottomY), fisheyeAngle, fisheyeCenter, dewarpingParameters);
        elevationTop = getElevationFromDewarpedImagePixel(Point<float>(rightX, topY), fisheyeAngle, fisheyeCenter, dewarpingParameters);
        azimuthLeft = getAzimuthFromDewarpedImagePixel(Point<float>(leftX, topY), fisheyeCenter, dewarpingParameters);
        azimuthRight = getAzimuthFromDewarpedImagePixel(Point<float>(rightX, bottomY), fisheyeCenter, dewarpingParameters);
    }

    float deltaAzimuthLeft = std::abs(azimuth - azimuthLeft);
    float deltaAzimuthRight = std::abs(azimuth - azimuthRight);
    float azimuthSpan = std::min(deltaAzimuthLeft, deltaAzimuthRight) * 2;

    float deltaElevationTop = std::abs(elevation - elevationTop);
    float deltaElevationBottom = std::abs(elevation - elevationBottom);
    float elevationSpan = deltaElevationTop + deltaElevationBottom;
    elevation = elevationTop - elevationSpan / 2.f;

    SphericalAngleRect sphericalRect(azimuth, elevation, azimuthSpan, elevationSpan);

    return sphericalRect;
}

LinearPixelFilter getLinearPixelFilter(const Point<float>& pixel, const Dim2<int>& dim)
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

float getElevationFromDewarpedImagePixel(const Point<float>& pixel, float fisheyeAngle, const Point<float>& fisheyeCenter,
                            const DewarpingParameters& dewarpingParameters)
{
    if (fisheyeAngle > 2.f * math::PI)
    {
        throw std::invalid_argument("Fisheye angle must be in radian!");
    }

    Point<float> normalizedPixel(pixel.x / dewarpingParameters.dewarpWidth, pixel.y / dewarpingParameters.dewarpHeight);
    Point<float> sourcePixel = getSourcePixelFromDewarpedImageNormalizedPixel(normalizedPixel, dewarpingParameters);
    float distanceToFisheyeCenter = math::euclideanDistance(Point<float>(sourcePixel.x - fisheyeCenter.x, sourcePixel.y - fisheyeCenter.y));
    float elevation = math::getElevationFromDistanceToFisheyeCenter(distanceToFisheyeCenter, fisheyeCenter.x, fisheyeAngle);

    return elevation;
}

float getAzimuthFromDewarpedImagePixel(const Point<float>& pixel, const Point<float>& fisheyeCenter,
                                       const DewarpingParameters& dewarpingParameters)
{
    Point<float> normalizedPixel(pixel.x / dewarpingParameters.dewarpWidth, pixel.y / dewarpingParameters.dewarpHeight);
    Point<float> sourcePixel = getSourcePixelFromDewarpedImageNormalizedPixel(normalizedPixel, dewarpingParameters);
    Point<float> distanceToFisheyeCenter(sourcePixel.x - fisheyeCenter.x, sourcePixel.y - fisheyeCenter.y);
    float azimuth = math::getAzimuthFromDistanceToFisheyeCenter(distanceToFisheyeCenter);

    return azimuth;
}

}    // namespace Model
