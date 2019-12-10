#ifndef VC_CAMERA_READER2_H
#define VC_CAMERA_READER2_H

#include <linux/videodev2.h>

#include "model/stream/utils/models/circular_buffer.h"
#include "model/stream/video/input/base_camera_reader.h"
#include "model/stream/video/video_config.h"
#include "model/stream/utils/alloc/heap_object_factory.h"

namespace Model
{
class VcCameraReader : public BaseCameraReader
{
   public:
    VcCameraReader(std::shared_ptr<VideoConfig> cameraConfig, std::size_t bufferCount);

    bool readImage(Image& image) override;

   protected:
    void initializeInternal() override;
    void finalizeInternal() override;

    CircularBuffer<Image> images_;

   private:
    HeapObjectFactory heapObjectFactory_;
};

}    // namespace Model

#endif    //! VC_CAMERA_READER2_H
