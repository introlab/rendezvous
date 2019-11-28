#ifndef IMAGE_DRAWING_H
#define IMAGE_DRAWING_H

#include "model/stream/utils/images/image_format.h"
#include "model/stream/utils/images/images.h"

namespace Model
{
class ImageDrawing
{
   public:
    static void drawBordersUYVY(Image& image, int borderWidth, RGB color);

   private:
    static void drawPixelUYVU(UYVY* image, int width, int xPixel, int yPixel, RGB color);
};
}    // namespace Model

#endif    // IMAGE_DRAWING_H
