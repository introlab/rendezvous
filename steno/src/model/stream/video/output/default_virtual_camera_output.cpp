#include "default_virtual_camera_output.h"

#include <QProcess>

namespace Model
{
/**
 * @brief Initialize the v4l2loopback device with the default image.
 * @param [IN] - deviceName
 */
void DefaultVirtualCameraOutput::writeDefaultImage(const QString& deviceName)
{
    QProcess process;
    process.execute(
        "ffmpeg -loglevel panic -re -i ../resources/defaultImage.jpg -f v4l2 -vcodec rawvideo -pix_fmt uyvy422 " +
        deviceName);

    process.waitForFinished();
}

}    // namespace Model
