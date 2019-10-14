#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

#include "streaming/input/IVideoStream.h"

class VideoStream : public IVideoStream
{
public:

    bool copyFrameData(const Image& image) override;
    bool getResolution(Dim3<int>& dim) override;

};

#endif // !VIDEO_STREAM_H
