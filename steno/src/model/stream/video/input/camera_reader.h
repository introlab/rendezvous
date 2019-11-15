#ifndef CAMERA_READER_H
#define CAMERA_READER_H

#include <linux/videodev2.h>

#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class CameraReader : public IVideoInput
{
   public:
    CameraReader(std::shared_ptr<VideoConfig> cameraConfig, std::size_t bufferCount);

    void open() override;
    void close() override;
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

    std::shared_ptr<VideoConfig> videoConfig_;
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

}    // namespace Model

#endif    //! CAMERA_READER_H
