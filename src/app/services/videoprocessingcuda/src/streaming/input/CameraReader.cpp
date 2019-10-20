#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "utils/images/stb/stb_image.h"
#include "CameraReader.h"
#include <iostream>

CameraReader::CameraReader() {}

CameraReader::CameraReader(CameraConfig cameraConfig)
        : cameraConfig_(cameraConfig)
{
    rgb = new unsigned char[cameraConfig_.resolution.width * cameraConfig_.resolution.height * cameraConfig_.resolution.channels];
}

CameraReader::~CameraReader() {
    stop();
}

void CameraReader::stop()
{
    close(fd);
}

void CameraReader::start()
{

    fd = open(cameraConfig_.deviceName.c_str(), O_RDWR);
    if (fd == -1)
    {
        perror("Opening video device");
        return;
    }
    if(print_caps(fd))
    {
        return;
    }
    if(init_mmap(fd))
    {
        perror("Error with init_mmap");
        return;
    }

}

unsigned char* CameraReader::readFrame()
{
    
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
    {
        perror("Query Buffer");
    }
 
    if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type))
    {
        perror("Start Capture");
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2;
    int r = select(fd+1, &fds, NULL, NULL, &tv);
    if(-1 == r)
    {
        perror("Waiting for Frame");
    }

    if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
    {
        perror("Retrieving Frame");
    }

    int width = cameraConfig_.resolution.width;
    int height = cameraConfig_.resolution.height;
    int channels = cameraConfig_.resolution.channels;
    int size = width * height * channels;

    convertToRGB(width, height, channels, buffer, rgb);

    return rgb;
}
 
int CameraReader::xioctl(int fd, int request, void *arg)
{
        int r;
 
        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
}
 
int CameraReader::init_mmap(int fd)
{
    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
 
    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        perror("Requesting Buffer");
        return 1;
    }

    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    
    if(-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
    {
        perror("Querying Buffer");
        return 1;
    }

    buffer = (uint8_t*) mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

    printf("Length: %d\nAddress: %p\n", buf.length, buffer);
    printf("Image Length: %d\n", buf.bytesused);
 
    return 0;
}

int CameraReader::clamp(int min, int max, int value)
{
    if (value < min)
        return min;
    if (value > max)
        return max;
    return value;
}

RGB CameraReader::getRGBPixel(const YUV422& yuv)
{
   RGB rgb;
   int c = yuv.y - 16;
   int d = yuv.u - 128;
   int e = yuv.v - 128;

   rgb.r = clamp(0, 255, (298 * c + 409 * e + 128) >> 8);
   rgb.g = clamp(0, 255, (298 * c - 100 * d - 208 * e + 128) >> 8);
   rgb.b = clamp(0, 255, (298 * c + 516 * d + 128) >> 8);

   return rgb;
}

void CameraReader::convertToRGB(int width, int height, int channels, uint8_t *yuv, uint8_t *rgb)
{
    int size = width * height * channels;
    for (int i = 0, j = 0; i < size; i += 6, j += 4)
    {
        YUV422 yuv1, yuv2;

        yuv1.u = yuv[j];
        yuv2.u = yuv1.u;

        yuv1.v = yuv[j + 2];
        yuv2.v = yuv1.v;

        yuv1.y = yuv[j + 1];
        yuv2.y = yuv[j + 3];

        RGB rgb1 = getRGBPixel(yuv1);
        RGB rgb2 = getRGBPixel(yuv2);

        rgb[i] = rgb1.r;
        rgb[i + 1] = rgb1.g;
        rgb[i + 2] = rgb1.b;
        rgb[i + 3] = rgb2.r;
        rgb[i + 4] = rgb2.g;
        rgb[i + 5] = rgb2.b;
    }
}

int CameraReader::print_caps(int fd)
{
    struct v4l2_capability caps = {};
        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
        {
                perror("Querying Capabilities");
                return 1;
        }
 
        printf( "Driver Caps:\n"
                "  Driver: \"%s\"\n"
                "  Card: \"%s\"\n"
                "  Bus: \"%s\"\n"
                "  Version: %d.%d\n"
                "  Capabilities: %08x\n",
                caps.driver,
                caps.card,
                caps.bus_info,
                (caps.version>>16)&&0xff,
                (caps.version>>24)&&0xff,
                caps.capabilities);
 
 
        struct v4l2_cropcap cropcap = {0};
        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
        {
                perror("Querying Cropping Capabilities");
                return 1;
        }
 
        printf( "Camera Cropping:\n"
                "  Bounds: %dx%d+%d+%d\n"
                "  Default: %dx%d+%d+%d\n"
                "  Aspect: %d/%d\n",
                cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
                cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
                cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);
 
        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        char fourcc[5] = {0};
        char c, e;
        printf("  FMT : CE Desc\n--------------------\n");
        
        while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
        {
                strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
                c = fmtdesc.flags & 1? 'C' : ' ';
                e = fmtdesc.flags & 2? 'E' : ' ';
                printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
                fmtdesc.index++;
        }
 
        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = 2880;
        fmt.fmt.pix.height = 2160;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUV420;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        
        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            perror("Setting Pixel Format");
            return 1;
        }
 
        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
        printf( "Selected Camera Mode:\n"
                "  Width: %d\n"
                "  Height: %d\n"
                "  PixFmt: %s\n"
                "  Field: %d\n",
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fourcc,
                fmt.fmt.pix.field);
        return 0;
}