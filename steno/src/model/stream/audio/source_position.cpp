#include "source_position.h"

#include "model/stream/utils/math/angle_calculations.h"

namespace Model
{
SourcePosition::SourcePosition(float azimuth, float elevation)
    : azimuth(azimuth)
    , elevation(elevation)
{
}

SourcePosition SourcePosition::deserialize(const QJsonValue& jsonSource)
{
    float x = static_cast<float>(jsonSource["x"].toDouble());
    float y = static_cast<float>(jsonSource["y"].toDouble());
    float z = static_cast<float>(jsonSource["z"].toDouble());
    float azimuth = math::getAzimuthFromPosition(x, y);
    float elevation = math::getElevationFromPosition(x, y, z);

    return SourcePosition(azimuth, elevation);
}

}    // namespace Model