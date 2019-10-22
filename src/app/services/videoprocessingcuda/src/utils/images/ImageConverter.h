#ifndef IMAGE_CONVERTER_H
#define IMAGE_CONVERTER_H

#include "utils/images/IImageConverter.h"
#include "utils/images/Images.h"

class ImageConverter : public IImageConverter
{
public:

    void convert(const Image& inImage, const Image& outImage) override;

private:

    void convert(const Dim2<int>& dim, const RGB* inData, UYVY* outData);
    void convert(const Dim2<int>& dim, const RGB* inData, YUYV* outData);
    void convert(const Dim2<int>& dim, const UYVY* inData, RGB* outData);
    void convert(const Dim2<int>& dim, const YUYV* inData, RGB* outData);
    void getRGBFromUYVY(const UYVY& uyvy, RGB& rgb1, RGB& rgb2);
    void getRGBFromYUYV(const YUYV& uyvy, RGB& rgb1, RGB& rgb2);
    void getUYVYFromRGB(const RGB& rgb1, const RGB& rgb2, UYVY& uyvy);
    void getYUYVFromRGB(const RGB& rgb1, const RGB& rgb2, YUYV& uyvy);

};

#endif //!IMAGE_CONVERTER_H