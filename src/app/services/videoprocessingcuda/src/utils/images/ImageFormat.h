#ifndef IMAGE_FORMAT_H
#define IMAGE_FORMAT_H

#include <string>

#include "utils/macros/Packing.h"

enum class ImageFormat
{
    RGB_FMT,
    UYVY_FMT,
    YUYV_FMT
};

PACK(struct RGB
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
});

PACK(struct UYVY
{
    unsigned char u;
    unsigned char y1;
    unsigned char v;
    unsigned char y2;
});

PACK(struct YUYV
{
    unsigned char y1;
    unsigned char u;
    unsigned char y2;
    unsigned char v;
});

std::string getImageFormatString(ImageFormat format);
unsigned int getV4L2Format(ImageFormat format);
std::string getV4L2FormatString(unsigned int format);
float getBytesPerPixel(ImageFormat format);

#endif //!IMAGE_FORMAT_H