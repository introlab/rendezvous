// This class uses libv4l2cpp
// Source: https://github.com/mpromonet/libv4l2cpp

#ifndef VIRTUAL_CAMERA_DEVICE_H
#define VIRTUAL_CAMERA_DEVICE_H

#include <list>

#include <V4l2Output.h>

enum ImageFormat
{
    UYVY,
    YUYV,
    YUV420
};

#ifndef RGB_YUV422
#define RGB_YUV422
struct RGB
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct YUV422 {
    unsigned char u;
    unsigned char y;
    unsigned char v;
};
#endif //!RGB_YUV422

class VirtualCameraDevice
{
public:
    VirtualCameraDevice(const char* videoDevice, ImageFormat format, unsigned int width, unsigned int height, unsigned int fps);
    ~VirtualCameraDevice();

    bool isWritable(timeval timeout);

    size_t write(unsigned char* buffer, int width, int height, int channels);

    void stopDevice();

private:
    V4l2Output* videoOutput = nullptr;

    void convertToYUV422(int width, int height, int channels, unsigned char* rgb, unsigned char* yuv422);

    YUV422 getYUV422(const RGB& rgb);
};

#endif //!VIRTUAL_CAMERA_DEVICE_H