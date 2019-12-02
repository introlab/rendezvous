#ifndef I_VIRTUAL_CAMERA_SOURCE_H
#define I_VIRTUAL_CAMERA_SOURCE_H

#include <vector>

#include "virtual_camera.h"

namespace Model
{
class IVirtualCameraSource
{
   public:
    virtual ~IVirtualCameraSource() = default;

    virtual std::vector<VirtualCamera> getVirtualCameras() = 0;
};

}    // namespace Model

#endif    // I_VIRTUAL_CAMERA_SOURCE_H
