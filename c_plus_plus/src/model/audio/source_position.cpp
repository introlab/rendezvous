#include "source_position.h"

#include "model/utils/spherical_angle_converter.h"


namespace Model 
{

SourcePosition::SourcePosition(double azimuth, double elevation) :
    azimuth(azimuth),
    elevation(elevation)
{
}

SourcePosition::~SourcePosition()
{
}

SourcePosition SourcePosition::deserialize(const QJsonValue jsonSource)
{
    double x = jsonSource["x"].toDouble();
    double y = jsonSource["y"].toDouble();
    double z = jsonSource["z"].toDouble();
    double azimuth   = SphericalAngleConverter::getAzimuthFromPosition(x, y);
    double elevation = SphericalAngleConverter::getElevationFromPosition(x, y, z);

    return SourcePosition(azimuth, elevation);
}

} // Model