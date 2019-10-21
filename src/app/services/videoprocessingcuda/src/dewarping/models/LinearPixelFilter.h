#ifndef LINEAR_PIXEL_FILTER_H
#define LINEAR_PIXEL_FILTER_H

struct PixelContribution
{
    int index;
    float ratio;
};

struct LinearPixelFilter
{
    PixelContribution pc1;
    PixelContribution pc2;
    PixelContribution pc3;
    PixelContribution pc4;
};

#endif // !LINEAR_PIXEL_FILTER_H