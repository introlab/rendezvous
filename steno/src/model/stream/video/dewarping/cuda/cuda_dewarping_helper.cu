#include "cuda_dewarping_helper.cuh"

#include "model/stream/utils/math/math_constants.h"

namespace Model
{

__device__ Point<float> getSourcePixelFromDewarpedImageNormalizedPixelDevice(const Point<float>& normalizedPixel, const DewarpingParameters& dewarpingParameters)
{
    float xRadiusFactor = __sinf(math::PI * normalizedPixel.x);
    float yRadiusFactor = __sinf((math::PI * normalizedPixel.y) / 2.f);

    float dewarpDimensionX = normalizedPixel.x * dewarpingParameters.dewarpWidth;
    float dewarpDimensionY = normalizedPixel.y * dewarpingParameters.dewarpHeight;

    float radius = dewarpDimensionY + dewarpingParameters.inRadius +
                   dewarpingParameters.outRadiusDiff * xRadiusFactor * yRadiusFactor *
                   dewarpingParameters.bottomDistorsionFactor;
    float theta = (dewarpDimensionX + dewarpingParameters.xOffset) / dewarpingParameters.centerRadius;

    Point<float> sourcePixel;
    sourcePixel.x = dewarpingParameters.xCenter + radius * __sinf(theta);
    sourcePixel.y = dewarpingParameters.yCenter + radius * __cosf(theta);

    return sourcePixel;
}

__device__ LinearPixelFilter calculateLinearPixelFilterDevice(const Point<float>& pixel, const Dim2<int>& dim)
{
    int xRoundDown = int(pixel.x);
    int yRoundDown = int(pixel.y);
    float xRatio = pixel.x - xRoundDown;
    float yRatio = pixel.y - yRoundDown;
    float xOpposite = 1 - xRatio;
    float yOpposite = 1 - yRatio;

    LinearPixelFilter linearPixelFilter;

    linearPixelFilter.pc1.index = (xRoundDown + (yRoundDown * dim.width)) * 3;
    linearPixelFilter.pc2.index = linearPixelFilter.pc1.index + 3;
    linearPixelFilter.pc3.index = linearPixelFilter.pc1.index + dim.width * 3;
    linearPixelFilter.pc4.index = linearPixelFilter.pc2.index + dim.width * 3;

    linearPixelFilter.pc1.ratio = xOpposite * yOpposite;
    linearPixelFilter.pc2.ratio = xRatio * yOpposite;
    linearPixelFilter.pc3.ratio = xOpposite * yRatio;
    linearPixelFilter.pc4.ratio = xRatio * yRatio;

    return linearPixelFilter;
}

__device__ Point<float> getNormalizedPixelFromIndexDevice(int index, const Dim2<int>& dim)
{
    Point<float> normalizedPoint;
    normalizedPoint.x = float(index % dim.width) / dim.width;
    normalizedPoint.y = float(index / dim.width) / dim.height;

    return normalizedPoint;
}

__device__ int getSourcePixelIndexDevice(const Point<float>& pixel, const Dim2<int>& dim)
{
    return (int(pixel.x) + int(pixel.y) * dim.width);
}

int calculateKernelBlockCount(const Dim2<int>& dim, int blockSize)
{
    return (dim.width * dim.height + blockSize - 1) / blockSize;
}

}    // namespace Model