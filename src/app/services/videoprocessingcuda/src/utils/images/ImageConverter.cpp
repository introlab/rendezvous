#include "ImageConverter.h"

#include <stdexcept>
#include <cstring>
#include <iostream>

#include "utils/math/Helpers.h"

void ImageConverter::convert(const Image& inImage, const Image& outImage)
{
    if (inImage.format == ImageFormat::RGB_FMT && outImage.format == ImageFormat::UYVY_FMT)
    {
        const RGB* rbgData = reinterpret_cast<const RGB*>(inImage.hostData);
        UYVY* uyvyData = reinterpret_cast<UYVY*>(outImage.hostData);
        convert(inImage, rbgData, uyvyData);
    }
    else if (inImage.format == ImageFormat::RGB_FMT && outImage.format == ImageFormat::YUYV_FMT)
    {
        const RGB* rbgData = reinterpret_cast<const RGB*>(inImage.hostData);
        YUYV* yuyvData = reinterpret_cast<YUYV*>(outImage.hostData);
        convert(inImage, rbgData, yuyvData);
    }
    else if (inImage.format == ImageFormat::UYVY_FMT && outImage.format == ImageFormat::RGB_FMT)
    {
        const UYVY* uyvyData = reinterpret_cast<const UYVY*>(inImage.hostData);
        RGB* rbgData = reinterpret_cast<RGB*>(outImage.hostData);
        convert(inImage, uyvyData, rbgData);
    }
    else if (inImage.format == ImageFormat::YUYV_FMT && outImage.format == ImageFormat::RGB_FMT)
    {
        const YUYV* yuyvData = reinterpret_cast<const YUYV*>(inImage.hostData);
        RGB* rbgData = reinterpret_cast<RGB*>(outImage.hostData);
        convert(inImage, yuyvData, rbgData);
    }
    else if (inImage.format == outImage.format)
    {
        std::memcpy(outImage.hostData, inImage.hostData, inImage.size);
        std::cout << "WARNING : Conversion from same formats, copying data instead" << std::endl;
    }
    else
    {
        throw std::invalid_argument("Conversion from " + getImageFormatString(inImage.format) + " to " + 
                                    getImageFormatString(outImage.format) + " is not defined!");
    }
}

void ImageConverter::convert(const Dim2<int>& dim, const RGB* inData, UYVY* outData)
{
    int size = dim.width * dim.height;

    for (int i = 0, j = 0; i < size; i += 2, ++j)
    {
        getUYVYFromRGB(inData[i], inData[i + 1], outData[j]);
    }
}

void ImageConverter::convert(const Dim2<int>& dim, const RGB* inData, YUYV* outData)
{
    int size = dim.width * dim.height;

    for (int i = 0, j = 0; i < size; i += 2, ++j)
    {
        getYUYVFromRGB(inData[i], inData[i + 1], outData[j]);
    }
}

void ImageConverter::convert(const Dim2<int>& dim, const UYVY* inData, RGB* outData)
{
    int size = dim.width * dim.height;

    for (int i = 0, j = 0; i < size; i += 2, ++j)
    {
        getRGBFromUYVY(inData[j], outData[i], outData[i + 1]);
    }
}

void ImageConverter::convert(const Dim2<int>& dim, const YUYV* inData, RGB* outData)
{
    int size = dim.width * dim.height;

    for (int i = 0, j = 0; i < size; i += 2, ++j)
    {
        getRGBFromYUYV(inData[j], outData[i], outData[i + 1]);
    }
}

void ImageConverter::getRGBFromUYVY(const UYVY& uyvy, RGB& rgb1, RGB& rgb2)
{
    int c1 = uyvy.y1 - 16;
    int c2 = uyvy.y2 - 16;
    int d = uyvy.u - 128;
    int e = uyvy.v - 128;

    rgb1.r = math::clamp((298 * c1 + 409 * e + 128) >> 8, 0, 255);
    rgb1.g = math::clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb1.b = math::clamp((298 * c1 + 516 * d + 128) >> 8, 0, 255);

    rgb2.r = math::clamp((298 * c2 + 409 * e + 128) >> 8, 0, 255);
    rgb2.g = math::clamp((298 * c2 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb2.b = math::clamp((298 * c2 + 516 * d + 128) >> 8, 0, 255);
}

void ImageConverter::getRGBFromYUYV(const YUYV& yuyv, RGB& rgb1, RGB& rgb2)
{
    int c1 = yuyv.y1 - 16;
    int c2 = yuyv.y2 - 16;
    int d = yuyv.u - 128;
    int e = yuyv.v - 128;

    rgb1.r = math::clamp((298 * c1 + 409 * e + 128) >> 8, 0, 255);
    rgb1.g = math::clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb1.b = math::clamp((298 * c1 + 516 * d + 128) >> 8, 0, 255);

    rgb2.r = math::clamp((298 * c2 + 409 * e + 128) >> 8, 0, 255);
    rgb2.g = math::clamp((298 * c2 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb2.b = math::clamp((298 * c2 + 516 * d + 128) >> 8, 0, 255);
}

void ImageConverter::getUYVYFromRGB(const RGB& rgb1, const RGB& rgb2, UYVY& uyvy)
{
    int r1 = (int)rgb1.r;
    int g1 = (int)rgb1.g;
    int b1 = (int)rgb1.b;
    int r2 = (int)rgb2.r;
    int g2 = (int)rgb2.g;
    int b2 = (int)rgb2.b;

    uyvy.u = ((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128;
    uyvy.y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    uyvy.v = ((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128;
    uyvy.y2 = ((66 * r2 + 129 * g2 + 25 * b2 + 128) >> 8) + 16;
}

void ImageConverter::getYUYVFromRGB(const RGB& rgb1, const RGB& rgb2, YUYV& yuyv)
{
    int r1 = (int)rgb1.r;
    int g1 = (int)rgb1.g;
    int b1 = (int)rgb1.b;
    int r2 = (int)rgb2.r;
    int g2 = (int)rgb2.g;
    int b2 = (int)rgb2.b;

    yuyv.u = ((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128;
    yuyv.y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    yuyv.v = ((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128;
    yuyv.y2 = ((66 * r2 + 129 * g2 + 25 * b2 + 128) >> 8) + 16;
}