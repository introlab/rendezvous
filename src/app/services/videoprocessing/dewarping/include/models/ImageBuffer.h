#ifndef IMAGE_BUFFER_H
#define IMAGE_BUFFER_H

struct ImageBuffer
{
    ImageBuffer();
    ImageBuffer(unsigned char * image, int width, int height, int channels);
    
    unsigned char * image;
    int width;
    int height;
    int channels;
    int size;
};

#endif //!IMAGE_BUFFER_H