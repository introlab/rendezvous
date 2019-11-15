#ifndef VIRTUAL_CAMERA_OUTPUT_H
#define VIRTUAL_CAMERA_OUTPUT_H

#include <3rd/v4l2/include/V4l2Output.h>

#include "model/stream/utils/models/dim2.h"
#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"

namespace Model
{
class VirtualCameraOutput : public IVideoOutput
{
   public:
    explicit VirtualCameraOutput(std::shared_ptr<VideoConfig> videoConfig);

    void open() override;
    void close() override;
    void writeImage(const Image& image) override;

   private:
    std::shared_ptr<VideoConfig> videoConfig_;
    V4l2Output* videoOutput_;
};

}    // namespace Model

#endif    // !VIRTUAL_CAMERA_OUTPUT_H
