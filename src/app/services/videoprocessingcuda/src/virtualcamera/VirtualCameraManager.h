#ifndef VIRTUAL_CAMERA_MANAGER_H
#define VIRTUAL_CAMERA_MANAGER_H

#include <vector>
#include <map>

#include "utils/models/AngleRect.h"
#include "virtualcamera/VirtualCamera.h"

class VirtualCameraManager
{
public:

    VirtualCameraManager(float aspectRatio, float srcImageMinElevation, float srcImageMaxElevation);

    void updateVirtualCameras(int elapsedTimeMs);
    void updateVirtualCamerasGoal(const std::vector<AngleRect>& goals);
    const std::vector<VirtualCamera>& getVirtualCameras();

private:

    float getElevationOverflow(float elevation, float elevationSpan);
    std::vector<AngleRect> getUniqueRegions(const std::vector<AngleRect>& regions);
    void updateRegionsBounds(std::vector<AngleRect>& regions);
    std::multimap<float, std::pair<int, int>> getVcToGoalOrderedDistances(const std::vector<AngleRect>& targets);

    float aspectRatio_;
    float srcImageMinElevation_;
    float srcImageMaxElevation_;
    float srcImageMaxElevationSpan_;

    std::vector<VirtualCamera> virtualCameras_;

};

#endif // !VIRTUAL_CAMERA_MANAGER_H
