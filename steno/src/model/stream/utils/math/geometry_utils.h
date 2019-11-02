#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#include "model/stream/utils/models/bounding_box.h"
#include "model/stream/utils/models/rectangle.h"
#include "model/stream/utils/models/spherical_angle_box.h"
#include "model/stream/utils/models/spherical_angle_rect.h"

namespace Model
{
namespace math
{
SphericalAngleRect convertToAngleRect(const SphericalAngleBox& angleBox);
SphericalAngleBox convertToAngleBox(const SphericalAngleRect& angleRect);

Rectangle convertToRectangle(const BoundingBox& boundingBox);
BoundingBox convertToBoundingBox(const Rectangle& rectangle);

}    // namespace math

}    // namespace Model

#endif    //! GEOMETRY_UTILS_H
