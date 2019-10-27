#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include "model/stream/utils/models/spherical_angle_rect.h"

namespace Model
{
struct VirtualCamera : SphericalAngleRect
{
    VirtualCamera(float azimuth, float elevation, float azimuthSpan, float elevationSpan, float timeToLiveMs)
        : SphericalAngleRect(azimuth, elevation, azimuthSpan, elevationSpan)
        , goal(azimuth, elevation, azimuthSpan, elevationSpan)
        , timeToLiveMs(timeToLiveMs)
    {
    }

    VirtualCamera(const SphericalAngleRect& region, float timeToLiveMs)
        : SphericalAngleRect(region)
        , goal(region)
        , timeToLiveMs(timeToLiveMs)
    {
    }

    SphericalAngleRect goal;

    int timeToLiveMs;
};

}    // namespace Model

#endif    // !VIRTUAL_CAMERA_H
