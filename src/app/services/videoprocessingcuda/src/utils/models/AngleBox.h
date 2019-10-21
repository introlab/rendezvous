#ifndef ANGLE_BOX_H
#define ANGLE_BOX_H

struct AngleBox
{
    AngleBox() = default;
    AngleBox(float leftAzimuth, float rightAzimuth, float bottomElevation, float topElevation)
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

#endif //!ANGLE_BOX_H