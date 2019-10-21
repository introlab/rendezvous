#ifndef CAMERA_READER_H
#define CAMERA_READER_H

 
#include "streaming/input/CameraConfig.h"

#ifndef RGB_YUV422
#define RGB_YUV422
struct RGB
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct YUV422
{
    unsigned char u;
    unsigned char y;
    unsigned char v;
};
#endif //!RGB_YUV422

class CameraReader
{
public:
    CameraReader();
    CameraReader(CameraConfig cameraConfig);
    ~CameraReader();

    void start();
    void stop();
    unsigned char* readFrame();
private:
    int fd;
    uint8_t* buffer;
    unsigned char* rgb;
    CameraConfig cameraConfig_;

    void convertToRGB(int width, int height, int channels, uint8_t *yuv, uint8_t *rgb);
    RGB getRGBPixel(const YUV422& yuv);
    int clamp(int min, int max, int value);
    int init_mmap(int fd);
    int xioctl(int fd, int request, void *arg);
    int print_caps(int fd);

};

#endif