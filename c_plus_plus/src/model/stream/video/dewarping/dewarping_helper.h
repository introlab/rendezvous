#ifndef DEWARPING_HELPER_H
#define DEWARPING_HELPER_H

#include <vector>

#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/dewarping/models/dewarping_parameters.h"
#include "model/stream/video/dewarping/models/donut_slice.h"
#include "model/stream/video/dewarping/models/linear_pixel_filter.h"
#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/utils/models/dim2.h"
#include "model/stream/utils/models/point.h"
#include "model/stream/utils/models/rectangle.h"

namespace Model
{

DonutSlice createDewarpingDonutSlice(DonutSlice& baseDonutSlice, float centersDistance);
DewarpingParameters getDewarpingParameters(const Dim2<int>& imageSize, const DewarpingConfig& dewarpingConfig, float middleAngle);
DewarpingParameters getDewarpingParameters(DonutSlice& baseDonutSlice, float topDistorsionFactor, float bottomDistorsionFactor);
DewarpingParameters getDewarpingParametersFromNewDonutSlice(DonutSlice& baseDonutSlice, DonutSlice& newDonutSlice,
                                                            float centersDistance, float bottomDistorsionFactor);
DewarpingParameters getDewarpingParametersFromAngleBoundingBox(const SphericalAngleRect& angleRect, const Point<float>& fisheyeCenter,
                                                               const DewarpingConfig& dewarpingConfig);
Point<float> getSourcePixelFromDewarpedImage(const Point<float>& pixel, const DewarpingParameters& dewarpingParameters);
SphericalAngleRect getAngleRectFromDewarpedImageRectangle(const Rectangle& rectangle, const DewarpingParameters& dewarpingParameters,
                                                 const Dim2<int>& imageSize, const Point<float>& fisheyeCenter, float fisheyeAngle);

Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index);
LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim2<int>& dim);
int calculateSourcePixelIndex(const Point<float>& pixel, const Dim2<int>& dim);
    
} // Model


#endif // !DEWARPING_HELPER_H
