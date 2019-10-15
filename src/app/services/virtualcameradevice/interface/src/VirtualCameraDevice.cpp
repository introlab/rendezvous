#include <VirtualCameraDevice.h>
#include "../../include/V4l2Device.h"

#include <string.h>
#include <iostream>

using namespace std;

VirtualCameraDevice::VirtualCameraDevice(const char* videoDevice, ImageFormat formatEnum, unsigned int width, unsigned int height, unsigned int fps)
{     

    unsigned int format;

    switch(formatEnum){
        case(ImageFormat::YUV420): format = V4L2_PIX_FMT_YUV420;
        break;

        case(ImageFormat::UYVY): format = V4L2_PIX_FMT_UYVY;
        break;

        case(ImageFormat::YUYV): format = V4L2_PIX_FMT_YUYV;
        break;
    }
    
    V4L2DeviceParameters param(videoDevice, format, width, height, fps);
    videoOutput = V4l2Output::create(param, V4l2Access::IOTYPE_READWRITE);
    
}

VirtualCameraDevice::~VirtualCameraDevice()
{
    videoOutput->stop();
}

bool VirtualCameraDevice::isWritable(timeval timeout)
{
    return videoOutput->isWritable(&timeout) == 1;
}

size_t VirtualCameraDevice::write(unsigned char * rgb, int width, int height, int channels)
{
    int rgbSize = width * height * channels;
    int yuvSize = 2 * rgbSize / 3;

    unsigned char yuv422[yuvSize];
    memset(yuv422, 0, yuvSize);

    convertToYUV422(width, height, channels, rgb, yuv422);

    size_t ret = videoOutput->write((char*)yuv422, yuvSize);

    return ret;
}

void VirtualCameraDevice::stopDevice() 
{
    videoOutput->stop();
}

YUV422 VirtualCameraDevice::getYUV422(const RGB& rgb)
{
    YUV422 yuv;
    int r = rgb.r;
    int g = rgb.g;
    int b = rgb.b;

    yuv.y = ((66 * b + 129 * g + 25 * r + 128) >> 8) + 16;
    yuv.u = ((-38 * b - 74 * g + 112 * r + 128) >> 8) + 128;
    yuv.v = ((112 * b - 94 * g - 18 * r + 128) >> 8) + 128;

    return yuv;
}

void VirtualCameraDevice::convertToYUV422(int width, int height, int channels, unsigned char* rgb, unsigned char* yuv422)
{
    int sizeRGB = width * height * channels;

    for (int i = 0, j = 0; i < sizeRGB; i += 6, j += 4)
    {
        RGB rgb1, rgb2;

        rgb1.r = rgb[i];
        rgb1.g = rgb[i + 1];
        rgb1.b = rgb[i + 2];
        rgb2.r = rgb[i + 3];
        rgb2.g = rgb[i + 4];
        rgb2.b = rgb[i + 5];

        YUV422 yuv1 = getYUV422(rgb1);
        YUV422 yuv2 = getYUV422(rgb2);

        yuv422[j] = yuv1.u;
        yuv422[j + 1] = yuv1.y;
        yuv422[j + 2] = yuv1.v;
        yuv422[j + 3] = yuv2.y;
    }
}