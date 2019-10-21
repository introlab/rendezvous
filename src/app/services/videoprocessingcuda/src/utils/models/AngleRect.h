#ifndef ANGLE_RECT_H
#define ANGLE_RECT_H

struct AngleRect
{
    AngleRect() = default;
    AngleRect(float azimuth, float elevation, float azimuthSpan, float elevationSpan)
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

#endif //!ANGLE_RECT_H