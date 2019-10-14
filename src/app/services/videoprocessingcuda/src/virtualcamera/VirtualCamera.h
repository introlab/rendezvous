#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include "utils/models/AngleRect.h"

struct VirtualCamera : AngleRect
{
    VirtualCamera(float azimuth, float elevation, float azimuthSpan, float elevationSpan, float timeToLiveMs)
        : AngleRect(azimuth, elevation, azimuthSpan, elevationSpan)
        , goal(azimuth, elevation, azimuthSpan, elevationSpan)
        , timeToLiveMs(timeToLiveMs)
    {
    }

    VirtualCamera(const AngleRect& region, float timeToLiveMs)
        : AngleRect(region)
        , goal(region)
        , timeToLiveMs(timeToLiveMs)
    {
    }

    AngleRect goal;

    float timeToLiveMs;
};

#endif // !VIRTUAL_CAMERA_H
