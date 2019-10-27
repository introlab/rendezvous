#ifndef VIRTUAL_CAMERA_MANAGER_H
#define VIRTUAL_CAMERA_MANAGER_H

#include <vector>
#include <map>

#include "model/stream/utils/models/spherical_angle_rect.h"
#include "model/stream/video/virtualcamera/virtual_camera.h"

namespace Model
{

class VirtualCameraManager
{
public:

    VirtualCameraManager(float aspectRatio, float srcImageMinElevation, float srcImageMaxElevation);

    void updateVirtualCameras(int elapsedTimeMs);
    void updateVirtualCamerasGoal(const std::vector<SphericalAngleRect>& goals);
    const std::vector<VirtualCamera>& getVirtualCameras();

private:

    float getElevationOverflow(float elevation, float elevationSpan);
    std::vector<SphericalAngleRect> getUniqueRegions(const std::vector<SphericalAngleRect>& regions);
    void updateRegionsBounds(std::vector<SphericalAngleRect>& regions);
    std::multimap<float, std::pair<int, int>> getVcToGoalOrderedDistances(const std::vector<SphericalAngleRect>& targets);

    float aspectRatio_;
    float srcImageMinElevation_;
    float srcImageMaxElevation_;
    float srcImageMaxElevationSpan_;

    std::vector<VirtualCamera> virtualCameras_;

};

} // Model

#endif // !VIRTUAL_CAMERA_MANAGER_H

