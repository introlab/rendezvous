#ifndef VIRTUAL_CAMERA_OUTPUT_H
#define VIRTUAL_CAMERA_OUTPUT_H

#include <V4l2Output.h>

#include "model/stream/video/output/i_video_output.h"
#include "model/stream/video/video_config.h"
#include "model/stream/utils/models/dim2.h"

namespace Model
{

class VirtualCameraOutput : public IVideoOutput
{
public:

    explicit VirtualCameraOutput(const VideoConfig& videoConfig);
    virtual ~VirtualCameraOutput();

    void writeImage(const Image& image) override;

private:

    VideoConfig videoConfig_;
    V4l2Output* videoOutput_;

};

} // Model

#endif // !VIRTUAL_CAMERA_OUTPUT_H

