#include "vc_camera_reader.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "model/stream/utils/time/time_utils.h"

namespace Model
{
VcCameraReader::VcCameraReader(std::shared_ptr<VideoConfig> videoConfig, std::size_t bufferCount)
    : BaseCameraReader(videoConfig)
    , images_(bufferCount,
              Image(videoConfig->resolution.width, videoConfig->resolution.height, videoConfig->imageFormat))
{
}

bool VcCameraReader::readImage(Image& image)
{
    image = images_.current();

    size_t size = read(fd_, image.hostData, image.size);

    if (size != image.size)
    {
        throw std::runtime_error("Could not read the entire image!");
    }

    images_.next();

    image.timeStamp = systemTimeSinceEpoch();

    return true;
}

void VcCameraReader::initializeInternal()
{
    heapObjectFactory_.allocateObjectCircularBuffer(images_);
}

void VcCameraReader::finalizeInternal()
{
    heapObjectFactory_.deallocateObjectCircularBuffer(images_);
}
}    // namespace Model
