#ifndef CAMERA_READER_H
#define CAMERA_READER_H

#include <linux/videodev2.h>

#include "stream/input/IVideoInput.h"
#include "stream/VideoConfig.h"
#include "utils/models/CircularBuffer.h"


class CameraReader : public IVideoInput
{
public:

    CameraReader(const VideoConfig& cameraConfig, std::size_t bufferCount);
    virtual ~CameraReader();

    const Image& readImage() override;

protected:

    struct IndexedImage
    {
        static int v4l2Index;

        IndexedImage() = default;
        IndexedImage(const Image& image);

        int index;
        Image image;
    };

    VideoConfig videoConfig_;
    CircularBuffer<IndexedImage> images_;

private:

    void queueCapture(v4l2_buffer& buf);
    void dequeueCapture(v4l2_buffer& buf);
    void requestBuffers(std::size_t bufferCount);
    void mapBuffer(IndexedImage& indexedImage);
    void unmapBuffer(IndexedImage& indexedImage);
    void checkCaps();
    void setImageFormat();
    int xioctl(int request, void* arg);

    v4l2_buffer buffer_;
    int fd_;

};

#endif //!CAMERA_READER_H