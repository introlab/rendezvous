#ifndef IMAGE_CONVERTER_H
#define IMAGE_CONVERTER_H

#include "model/stream/utils/images/i_image_converter.h"

namespace Model
{
class ImageConverter : public IImageConverter
{
   public:
    void convert(const Image& inImage, Image& outImage) override;

   private:
    void getRGBFromUYVY(const UYVY& uyvy, RGB& rgb1, RGB& rgb2);
    void getRGBFromYUYV(const YUYV& uyvy, RGB& rgb1, RGB& rgb2);
    void getUYVYFromRGB(const RGB& rgb1, const RGB& rgb2, UYVY& uyvy);
    void getYUYVFromRGB(const RGB& rgb1, const RGB& rgb2, YUYV& uyvy);
};

}    // namespace Model

#endif    //! IMAGE_CONVERTER_H
