#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#include "utils/models/AngleBox.h"
#include "utils/models/AngleRect.h"
#include "utils/models/BoundingBox.h"
#include "utils/models/Rectangle.h"

namespace math
{

AngleRect convertToAngleRect(const AngleBox& angleBox);
AngleBox convertToAngleBox(const AngleRect& angleRect);

Rectangle convertToRectangle(const BoundingBox& boundingBox);
BoundingBox convertToBoundingBox(const Rectangle& rectangle);

}

#endif //!GEOMETRY_UTILS_H