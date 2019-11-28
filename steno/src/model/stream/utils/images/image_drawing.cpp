#include "image_drawing.h"

namespace Model
{
void ImageDrawing::drawBorders(Image& image, ImageFormat imageFormat, int borderWidth, RGB color)
{
    UYVY* uyvyData;
    switch (imageFormat)
    {
        case ImageFormat::UYVY_FMT:
            uyvyData = reinterpret_cast<UYVY*>(image.hostData);
    }

    for (int xPixel = 0; xPixel < image.width / 2; xPixel++)
    {
        for (int yPixel = 0; yPixel < image.height; yPixel++)
        {
            // top border
            if (yPixel < borderWidth * 2)
            {
                drawPixel(uyvyData, imageFormat, image.width, xPixel, yPixel, color);
            }

            // bottom border
            if (yPixel >= image.height - borderWidth * 2)
            {
                drawPixel(uyvyData, imageFormat, image.width, xPixel, yPixel, color);
            }

            // left border
            if (xPixel < borderWidth)
            {
                drawPixel(uyvyData, imageFormat, image.width, xPixel, yPixel, color);
            }

            // right border
            if (xPixel >= image.width / 2 - borderWidth)
            {
                drawPixel(uyvyData, imageFormat, image.width, xPixel, yPixel, color);
            }
        }
    }
}

void ImageDrawing::drawPixel(UYVY* image, ImageFormat imageFormat, int width, int xPixel, int yPixel, RGB color)
{
    switch (imageFormat)
    {
        case ImageFormat::UYVY_FMT:
            int index = (width * yPixel / 2 + xPixel);
            image[index].u = ((-38 * color.r - 74 * color.g + 112 * color.b + 128) >> 8) + 128;
            image[index].y1 = ((66 * color.r + 129 * color.g + 25 * color.b + 128) >> 8) + 16;
            image[index].v = ((112 * color.r - 94 * color.g - 18 * color.b + 128) >> 8) + 128;
            image[index].y2 = ((66 * color.r + 129 * color.g + 25 * color.b + 128) >> 8) + 16;
            break;
    }
}
}    // namespace Model
