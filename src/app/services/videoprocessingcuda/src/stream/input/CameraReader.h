#ifndef CAMERA_READER_H
#define CAMERA_READER_H

#include <linux/videodev2.h>

#include "stream/input/CameraConfig.h"
#include "stream/input/IVideoInput.h"
#include "utils/models/TripleBuffer.h"


class CameraReader : public IVideoInput
{
public:

    CameraReader(const CameraConfig& cameraConfig);
    virtual ~CameraReader();

    const Image& readImage() override;

private:

    struct IndexedImage
    {
        static int v4l2Index;

        IndexedImage(const Image& image);

        int index;
        Image image;
    };

    void queueCapture(v4l2_buffer& buf);
    void dequeueCapture(v4l2_buffer& buf);
    void requestBuffers();
    void mapBuffer(IndexedImage& indexedImage);
    void unmapBuffer(IndexedImage& indexedImage);
    void checkCaps();
    void setImageFormat();
    int xioctl(int request, void* arg);

    CameraConfig cameraConfig_;
    TripleBuffer<IndexedImage> images_;
    v4l2_buffer buffer_;
    int fd_;

};

#endif //!CAMERA_READER_H