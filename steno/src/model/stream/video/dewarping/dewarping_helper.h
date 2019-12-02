#ifndef DEWARPING_HELPER_H
#define DEWARPING_HELPER_H

#include <vector>
#include <memory>

#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/dim2.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/rectangle.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/video/dewarping/models/donut_slice.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"

namespace Model
{
DonutSlice createDewarpingDonutSlice(const DonutSlice& baseDonutSlice, float centersDistance);
DewarpingParameters getDewarpingParameters(DonutSlice& baseDonutSlice, float topDistorsionFactor,
                                           float bottomDistorsionFactor);
DewarpingParameters getDewarpingParametersFromNewDonutSlice(DonutSlice& baseDonutSlice, DonutSlice& newDonutSlice,
                                                            float centersDistance, float bottomDistorsionFactor);

DewarpingParameters getDewarpingParametersFromSphericalAngleRect(const SphericalAngleRect& angleRect, 
                                                                 const DewarpingConfig& dewarpingConfig, 
                                                                 const Point<float>& fisheyeCenter);
bool isInOverlappingZone(const SphericalAngleRect& sphericalAngleRect, float angleSpan, float middleAngle);
SphericalAngleRect getAngleRectFromDewarpedImageRectangle(const Rectangle& rectangle,
                                                          const DewarpingParameters& dewarpingParameters,
                                                          const Dim2<int>& imageSize, const Point<float>& fisheyeCenter,
                                                          float fisheyeAngle);

int getSourcePixelIndex(const Point<float>& pixel, const Dim2<int>& dim);
Point<float> getNormalizedPixelFromIndex(int index, const Dim2<int>& dim);
Point<float> getSourcePixelFromDewarpedImageNormalizedPixel(const Point<float>& normalizedPixel, 
                                                            const DewarpingParameters& dewarpingParameters);
LinearPixelFilter getLinearPixelFilter(const Point<float>& pixel, const Dim2<int>& dim);

float getElevationFromDewarpedImagePixel(const Point<float>& pixel, float fisheyeAngle, const Point<float>& fisheyeCenter,
                                         const DewarpingParameters& dewarpingParameters);
float getAzimuthFromDewarpedImagePixel(const Point<float>& pixel, const Point<float>& fisheyeCenter,
                                       const DewarpingParameters& dewarpingParameters);

}    // namespace Model

#endif    // !DEWARPING_HELPER_H
