#include <models/ImageBuffer.h>

ImageBuffer::ImageBuffer()
   : image(0),
   width(0),
   height(0),
   channels(0),
   size(0)
{
}

ImageBuffer::ImageBuffer(unsigned char * image, int width, int height, int channels)
   : image(image),
   width(width),
   height(height),
   channels(channels),
   size(width * height * channels)
{
}