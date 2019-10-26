#ifndef SPHERICALANGLECONVERTER_H
#define SPHERICALANGLECONVERTER_H

namespace Model
{
class SphericalAngleConverter
{
   public:
    static double getAzimuthFromPosition(double x, double y);
    static double getElevationFromPosition(double x, double y, double z);
};

}    // Model

#endif    // SPHERICALANGLECONVERTER_H
