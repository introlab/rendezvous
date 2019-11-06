#include "default_virtual_camera_output.h"
#include "model/app_constants.h"

#include <QProcess>

namespace Model
{

void DefaultVirtualCameraOutput::writeDefaultImage()
{
    QProcess process;
    process.execute("ffmpeg -loglevel panic -re -i ../resources/defaultImage.jpg -f v4l2 -vcodec rawvideo -pix_fmt yuv420p " + Model::VIRTUAL_CAMERA_DEVICE);
}

} // Model
