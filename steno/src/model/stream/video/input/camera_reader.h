#ifndef CAMERA_READER_H
#define CAMERA_READER_H

#include <linux/videodev2.h>

#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/video/input/base_camera_reader.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class CameraReader : public BaseCameraReader
{
   public:
    CameraReader(std::shared_ptr<VideoConfig> cameraConfig, std::size_t bufferCount);

    void open() override;
    const Image& readImage() override;

   protected:
    void initializeInternal() override;
    void finalizeInternal() override;

    struct IndexedImage
    {
        static int v4l2Index;

        IndexedImage() = default;
        IndexedImage(const Image& image);

        int index;
        Image image;
    };

    CircularBuffer<IndexedImage> images_;

   private:
    void queueCapture(IndexedImage& indexedImage);
    void dequeueCapture(const IndexedImage& indexedImage);
    void requestBuffers(std::size_t bufferCount);
    void mapBuffer(IndexedImage& indexedImage);
    void unmapBuffer(IndexedImage& indexedImage);
};

}    // namespace Model

#endif    //! CAMERA_READER_H
