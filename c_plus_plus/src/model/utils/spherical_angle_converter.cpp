#include <cmath>

#include "spherical_angle_converter.h"

namespace Model
{
double SphericalAngleConverter::getAzimuthFromPosition(double x, double y)
{
    double tanRes = atan2(y, x);
    return std::fmod(tanRes, 2 * M_PI);
}

double SphericalAngleConverter::getElevationFromPosition(double x, double y, double z)
{
    double xyHypotenuse = sqrt(pow(y, 2) + pow(x, 2));
    double tanRes = atan2(z, xyHypotenuse);
    return std::fmod(tanRes, 2 * M_PI);
}

}    // namespace Model