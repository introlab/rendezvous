#ifndef VIRTUAL_CAMERA_MANAGER_H
#define VIRTUAL_CAMERA_MANAGER_H

#include <map>
#include <mutex>
#include <vector>

#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/video/virtualcamera/i_virtual_camera_source.h"
#include "model/stream/video/virtualcamera/virtual_camera.h"

namespace Model
{
class VirtualCameraManager : public IVirtualCameraSource
{
   public:
    VirtualCameraManager(float aspectRatio, float srcImageMinElevation, float srcImageMaxElevation);

    void updateVirtualCameras(int elapsedTimeMs);
    void updateVirtualCamerasGoal(const std::vector<SphericalAngleRect>& goals);
    void clearVirtualCameras();

    std::vector<VirtualCamera> getVirtualCameras() override;

   private:
    float getElevationOverflow(float elevation, float elevationSpan);
    std::vector<SphericalAngleRect> getUniqueRegions(const std::vector<SphericalAngleRect>& regions);
    void updateRegionsBounds(std::vector<SphericalAngleRect>& regions);
    std::multimap<float, std::pair<int, int>> getVcToGoalOrderedDistances(
        const std::vector<SphericalAngleRect>& targets);
    void setVirtualCameras(const std::vector<VirtualCamera>& virtualCameras);

    float aspectRatio_;
    float srcImageMinElevation_;
    float srcImageMaxElevation_;
    float srcImageMaxElevationSpan_;

    std::mutex mutex_;

    std::vector<VirtualCamera> virtualCameras_;
};

}    // namespace Model

#endif    // !VIRTUAL_CAMERA_MANAGER_H
