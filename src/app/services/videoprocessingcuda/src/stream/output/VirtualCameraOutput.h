#ifndef VIRTUAL_CAMERA_OUTPUT_H
#define VIRTUAL_CAMERA_OUTPUT_H

#include <V4l2Output.h>

#include "stream/output/IVideoOutput.h"
#include "stream/VideoConfig.h"
#include "utils/models/Dim2.h"

class VirtualCameraOutput : public IVideoOutput
{
public:

    VirtualCameraOutput(const VideoConfig& videoConfig);
    virtual ~VirtualCameraOutput();

    void writeImage(const Image& image) override;

private:

    VideoConfig videoConfig_;
    V4l2Output* videoOutput_;

};

#endif // !VIRTUAL_CAMERA_OUTPUT_H
