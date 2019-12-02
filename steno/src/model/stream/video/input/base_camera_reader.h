#ifndef BASE_CAMERA_READER_H
#define BASE_CAMERA_READER_H

#include <linux/videodev2.h>

#include "model/stream/video/input/i_video_input.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class BaseCameraReader : public IVideoInput
{
   public:
    BaseCameraReader(std::shared_ptr<VideoConfig> cameraConfig);

    void open() override;
    void close() override;

   protected:
    virtual void initializeInternal() {};
    virtual void finalizeInternal() {};
    void checkCaps();
    void setImageFormat();
    int xioctl(int request, void* arg);

    std::shared_ptr<VideoConfig> videoConfig_;
    v4l2_buffer buffer_;
    int fd_;
};

}    // namespace Model

#endif    //! BASE_CAMERA_READER_H
