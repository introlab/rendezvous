#ifndef SPHERICAL_ANGLE_RECT_H
#define SPHERICAL_ANGLE_RECT_H

namespace Model
{

struct SphericalAngleRect
{
    SphericalAngleRect() = default;
    SphericalAngleRect(float azimuth, float elevation, float azimuthSpan, float elevationSpan)
        : azimuth(azimuth)
        , elevation(elevation)
        , azimuthSpan(azimuthSpan)
        , elevationSpan(elevationSpan)
    {
    }

    float azimuth;
    float elevation;
    float azimuthSpan;
    float elevationSpan;
};

} // Model

#endif //!SPHERICAL_ANGLE_RECT_H
