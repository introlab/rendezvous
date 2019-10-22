#ifndef VIRTUAL_CAMERA_OUTPUT_H
#define VIRTUAL_CAMERA_OUTPUT_H

#include <V4l2Output.h>

#include "stream/output/IVideoOutput.h"
#include "utils/models/Dim2.h"

class VirtualCameraOutput : public IVideoOutput
{
public:

    VirtualCameraOutput(const std::string& videoDevice, const Dim2<int>& dim, ImageFormat format, unsigned int fps);
    virtual ~VirtualCameraOutput();

    void writeImage(const Image& image) override;

private:

    V4l2Output* videoOutput_;
    ImageFormat format_;

};

#endif // !VIRTUAL_CAMERA_OUTPUT_H
