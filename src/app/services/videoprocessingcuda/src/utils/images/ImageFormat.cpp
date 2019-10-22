#include "ImageFormat.h"

#include <cstring>
#include <stdexcept>

#include <linux/videodev2.h>

std::string getImageFormatString(ImageFormat format)
{
    switch (format)
    {
        case ImageFormat::RGB_FMT: return "RGB";
        case ImageFormat::UYVY_FMT: return "UYVY";
        case ImageFormat::YUYV_FMT: return "YUYV";
        default: return "Unknown format";
    }
}

unsigned int getV4L2Format(ImageFormat format)
{
    unsigned int v4l2Format;
    
    switch (format)
    {
    case ImageFormat::UYVY_FMT: 
        v4l2Format = V4L2_PIX_FMT_UYVY;
        break;
    case ImageFormat::YUYV_FMT: 
        v4l2Format = V4L2_PIX_FMT_YUYV;
        break;
    case ImageFormat::RGB_FMT:
        v4l2Format = V4L2_PIX_FMT_RGB24;
        break;
    default:
        throw std::invalid_argument("Undefined image format");
    }

    return v4l2Format;
}

std::string getV4L2FormatString(unsigned int format)
{
    char fourcc[5] = {};
    std::strncpy(fourcc, (char *)&format, 4);
    return std::string(fourcc);
}

float getBytesPerPixel(ImageFormat format)
{
    float bytesPerPixel;
    
    switch (format)
    {
    case ImageFormat::UYVY_FMT: 
        bytesPerPixel = 2.f;
        break;
    case ImageFormat::YUYV_FMT: 
        bytesPerPixel = 2.f;
        break;
    case ImageFormat::RGB_FMT:
        bytesPerPixel = 3.f;
        break;
    default:
        throw std::invalid_argument("Undefined image format");
    }

    return bytesPerPixel;
}