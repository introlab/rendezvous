#ifndef DEWARPING_HELPER_H
#define DEWARPING_HELPER_H

#include <vector>

#include "dewarping/models/DewarpingConfig.h"
#include "dewarping/models/DewarpingParameters.h"
#include "dewarping/models/DonutSlice.h"
#include "dewarping/models/LinearPixelFilter.h"
#include "utils/images/Image.h"
#include "utils/models/AngleRect.h"
#include "utils/models/Dim2.h"
#include "utils/models/Point.h"
#include "utils/models/Rectangle.h"

namespace dewarping
{

DonutSlice createDewarpingDonutSlice(DonutSlice& baseDonutSlice, float centersDistance);
DewarpingParameters getDewarpingParameters(const Dim2<int>& imageSize, const DewarpingConfig& dewarpingConfig, float middleAngle);
DewarpingParameters getDewarpingParameters(DonutSlice& baseDonutSlice, float topDistorsionFactor, float bottomDistorsionFactor);
DewarpingParameters getDewarpingParametersFromNewDonutSlice(DonutSlice& baseDonutSlice, DonutSlice& newDonutSlice,
                                                            float centersDistance, float bottomDistorsionFactor);
DewarpingParameters getDewarpingParametersFromAngleBoundingBox(const AngleRect& angleRect, const Point<float>& fisheyeCenter,
                                                               const DewarpingConfig& dewarpingConfig);
Point<float> getSourcePixelFromDewarpedImage(const Point<float>& pixel, const DewarpingParameters& dewarpingParameters);
AngleRect getAngleRectFromDewarpedImageRectangle(const Rectangle& rectangle, const DewarpingParameters& dewarpingParameters,
                                                 const Dim2<int>& imageSize, const Point<float>& fisheyeCenter, float fisheyeAngle);

Point<float> calculateSourcePixelPosition(const Dim2<int>& dst, const DewarpingParameters& params, int index);
LinearPixelFilter calculateLinearPixelFilter(const Point<float>& pixel, const Dim3<int>& dim);
int calculateSourcePixelIndex(const Point<float>& pixel, const Dim3<int>& dim);

}

#endif // !DEWARPING_HELPER_H