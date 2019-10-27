#ifndef IMAGES_H
#define IMAGES_H

#include <cstdint>
#include <string>

#include "model/stream/utils/models/dim2.h"
#include "model/stream/utils/images/image_format.h"

namespace Model
{

template<typename T>
struct ImageTemplate : public Dim2<int>
{
    ImageTemplate() = default;
    ImageTemplate(int width, int height, ImageFormat format)
        : Dim2<int>(width, height)
        , format(format)
        , bytesPerPixel(getBytesPerPixel(format))
        , size(width * height * bytesPerPixel)
        , hostData(nullptr)
        , deviceData(nullptr)
    {
    }

    ImageFormat format;
    float bytesPerPixel;
    std::size_t size;

    T* hostData;
    T* deviceData;
};

struct Image : public ImageTemplate<unsigned char>
{
    Image() = default;
    Image(int width, int height, ImageFormat format)
        : ImageTemplate<unsigned char>(width, height, format)
    {
    }

    Image(const Dim2<int>& dim, ImageFormat format)
        : Image(dim.width, dim.height, format)
    {
    }
};

struct ImageFloat : public ImageTemplate<float>
{
    ImageFloat(int width, int height, ImageFormat format)
        : ImageTemplate<float>(width, height, format)
    {
    }

    ImageFloat(const Dim2<int>& dim, ImageFormat format)
        : ImageFloat(dim.width, dim.height, format)
    {
    }
};

struct RGBImage : public Image
{
    RGBImage(int width, int height)
        :  Image(width, height, ImageFormat::RGB_FMT)
    {
    }

    explicit RGBImage(const Dim2<int>& dim)
        : RGBImage(dim.width, dim.height)
    {
    }
};

struct RGBImageFloat : public ImageFloat
{
    RGBImageFloat(int width, int height)
        :  ImageFloat(width, height, ImageFormat::RGB_FMT)
    {
    }

    explicit RGBImageFloat(const Dim2<int>& dim)
        : RGBImageFloat(dim.width, dim.height)
    {
    }
};

struct UYVYImage : public Image
{
    UYVYImage(int width, int height)
        :  Image(width, height, ImageFormat::UYVY_FMT)
    {
    }

    explicit UYVYImage(const Dim2<int>& dim)
        : UYVYImage(dim.width, dim.height)
    {
    }
};

struct YUYVImage : public Image
{
    YUYVImage(int width, int height)
        :  Image(width, height, ImageFormat::YUYV_FMT)
    {
    }

    explicit YUYVImage(const Dim2<int>& dim)
        : YUYVImage(dim.width, dim.height)
    {
    }
};

} // Model

#endif //!IMAGES_H

