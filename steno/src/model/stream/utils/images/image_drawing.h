#ifndef IMAGE_DRAWING_H
#define IMAGE_DRAWING_H

#include "model/stream/utils/images/image_format.h"
#include "model/stream/utils/images/images.h"

namespace Model
{
class ImageDrawing
{
   public:
    static void drawBorders(Image& image, ImageFormat imageFormat, int borderWidth, RGB color);

   private:
    static void drawPixel(UYVY* image, ImageFormat imageFormat, int width, int xPixel, int yPixel, RGB color);
};
}    // namespace Model

#endif    // IMAGE_DRAWING_H
