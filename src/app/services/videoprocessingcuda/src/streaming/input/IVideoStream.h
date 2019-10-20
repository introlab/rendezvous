#ifndef I_VIDEO_STREAM_H
#define I_VIDEO_STREAM_H

#include "utils/images/Image.h"
#include "utils/models/Dim3.h"

class IVideoStream
{
public:

    virtual ~IVideoStream() {};
    virtual bool copyFrameData(const Image& image) = 0;
    virtual bool getResolution(Dim3<int>& dim) = 0;

};

#endif // !I_VIDEO_STREAM_H
