// This class uses libv4l2cpp
// Source: https://github.com/mpromonet/libv4l2cpp

#include "../../include/V4l2Output.h"
#include <list>

using namespace std;

enum ImageFormat
{
    UYVY,
    YUV420
};

class VirtualCameraDevice
{
public:
    VirtualCameraDevice(const char* videoDevice, ImageFormat format, unsigned int width, unsigned int height, unsigned int fps);
    ~VirtualCameraDevice();

    bool isWritable(timeval timeout);

    size_t write(char* buffer, size_t bufferSize);

    void test();

    void stopDevice();

    //char convertToFormat(char* buffer, const unsigned int format);

private:
    V4l2Output* videoOutput = nullptr;
    unsigned int width;
    unsigned int height;

};