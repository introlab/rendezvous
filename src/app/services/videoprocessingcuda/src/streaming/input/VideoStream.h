#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

#include "streaming/input/IVideoStream.h"
#include "CameraConfig.h"
#include "CameraReader.h"

class VideoStream : public IVideoStream
{
public:
    VideoStream(CameraConfig cameraConfig);
    virtual ~VideoStream();

    bool copyFrameData(const Image& image) override;
    bool getResolution(Dim3<int>& dim) override;

private:
    Image image_;
    CameraReader cameraReader_;
};

#endif // !VIDEO_STREAM_H
