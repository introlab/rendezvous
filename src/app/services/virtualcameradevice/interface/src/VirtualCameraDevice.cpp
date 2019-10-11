#include <VirtualCameraDevice.h>
#include "../../include/V4l2Device.h"

#include <string.h>
#include <iostream>

#define ROUND_UP_2(num) (((num) + 1) & ~1)

using namespace std;

VirtualCameraDevice::VirtualCameraDevice(const char* videoDevice, ImageFormat formatEnum, unsigned int w, unsigned int h, unsigned int fps)
{     
    width = w;
    height = h;

    unsigned int format;
    if(formatEnum == ImageFormat::UYVY)
    {
        format = V4L2_PIX_FMT_UYVY;
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

size_t VirtualCameraDevice::write(char* buffer, size_t bufferSize)
{
    return videoOutput->write(buffer, bufferSize);
}

void VirtualCameraDevice::test()
{
    // Displays a green screen
    
    timeval timeout;
    bool isWritable = (VirtualCameraDevice::isWritable(timeout) == 1);

    size_t linewidth = (ROUND_UP_2(width) * 2);
    size_t framewidth = linewidth * height;
    char buffer[framewidth];

    cout << endl << "WIDTH = " << width << endl;
    cout << "HEIGHT = " << height << endl;
    cout << "LINEWIDTH = " << linewidth << endl;
    cout << "FRAMEWIDTH = " << framewidth << endl;

    memset(buffer, 0, framewidth);

    size_t nb = VirtualCameraDevice::write(buffer, framewidth);
}

void VirtualCameraDevice::stopDevice() 
{
    videoOutput->stop();
}

/*char VirtualCameraDevice::convertToFormat(char* buffer, const unsigned int format, const unsigned int width, const unsigned int height)
{
    size_t linewidth = 0;
    size_t framewidth = 0;

    if(format == V4L2_PIX_FMT_YUV420) 
    {
        linewidth = ROUND_UP_2(width) * 2;
        framewidth = linewidth * height; 
    }

    char convertedBuffer[framewidth];
    // do conversion

    return convertedBuffer;

}*/