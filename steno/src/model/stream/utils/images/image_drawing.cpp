#include "image_drawing.h"

namespace Model
{
/**
 * @brief Draws a border around the image received as a parameter
 * @param image - the image that we want to draw a border on
 * @param imageFormat - the format of the image (ex. UYVY)
 * @param borderWidth - the width of the border
 * @param color - the color of the border
 */
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

/**
 * @brief Draws a color at a specific pixel of an image
 * @param image - the image that we want to draw a pixel on
 * @param imageFormat - the format of the image (ex. UYVY)
 * @param imageWidth - the width of the image
 * @param xPixel - the x position of the pixel on the image
 * @param yPixel - the y position of the pixel on the image
 * @param color - the color of the pixel
 */
void ImageDrawing::drawPixel(UYVY* image, ImageFormat imageFormat, int imageWidth, int xPixel, int yPixel, RGB color)
{
    switch (imageFormat)
    {
        case ImageFormat::UYVY_FMT:
            int index = (imageWidth * yPixel / 2 + xPixel);
            image[index].u = ((-38 * color.r - 74 * color.g + 112 * color.b + 128) >> 8) + 128;
            image[index].y1 = ((66 * color.r + 129 * color.g + 25 * color.b + 128) >> 8) + 16;
            image[index].v = ((112 * color.r - 94 * color.g - 18 * color.b + 128) >> 8) + 128;
            image[index].y2 = ((66 * color.r + 129 * color.g + 25 * color.b + 128) >> 8) + 16;
            break;
    }
}
}    // namespace Model
