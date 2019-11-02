#ifndef SPHERICAL_ANGLE_BOX_H
#define SPHERICAL_ANGLE_BOX_H

namespace Model
{
struct SphericalAngleBox
{
    SphericalAngleBox() = default;
    SphericalAngleBox(float leftAzimuth, float rightAzimuth, float bottomElevation, float topElevation)
        : leftAzimuth(leftAzimuth)
        , rightAzimuth(rightAzimuth)
        , bottomElevation(bottomElevation)
        , topElevation(topElevation)
    {
    }

    float leftAzimuth;
    float rightAzimuth;
    float bottomElevation;
    float topElevation;
};

}    // namespace Model

#endif    //! SPHERICAL_ANGLE_BOX_H
