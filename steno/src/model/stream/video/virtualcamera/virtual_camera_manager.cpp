#include "virtual_camera_manager.h"

#include <cmath>
#include <limits>

#include "model/stream/utils/math/angle_calculations.h"
#include "model/stream/utils/math/helpers.h"
#include "model/stream/utils/math/math_constants.h"
#include "model/stream/utils/vector_utils.h"

namespace Model
{
namespace
{
const int VIRTUAL_CAMERA_TIME_TO_LIVE_MS = 5000;    // Time in seconds the virtual camera will remain if not updated
const float VIRTUAL_CAMERA_MIN_ELEVATION_SPAN = 0.3f;    // Minimum virtual camera's elevation span
const float NEW_VIRTUAL_CAMERA_CREATION_THRESHOLD =
    0.5f;                                           // Distance at which a new vc will be created instead of moved
const float POSITION_CHANGED_THRESHOLD = 0.03f;     // Change in position that cause a move of the virtual camera
const float DIMENTION_CHANGED_THRESHOLD = 0.10f;    // Change in dimension that cause a resize of the virtual camera
const float DUPLICATE_FACE_ANGLE_RANGE = 0.4f;      // Range in angles where we consider faces to be duplicate
const float ELEVATION_SHIFT_RATIO = 4.f;            // Shifting of elevation is required to center the virtual camera
const int TIME_TO_GOAL_MS = 850;                   // How much time is required for the camera to reach it's goal
}    // namespace

VirtualCameraManager::VirtualCameraManager(float aspectRatio, float srcImageMinElevation, float srcImageMaxElevation)
    : aspectRatio_(aspectRatio)
    , srcImageMinElevation_(srcImageMinElevation)
    , srcImageMaxElevation_(srcImageMaxElevation)
    , srcImageMaxElevationSpan_(srcImageMaxElevation - srcImageMinElevation)
{
}

void VirtualCameraManager::updateVirtualCameras(int elapsedTimeMs)
{
    std::vector<VirtualCamera> virtualCameras = getVirtualCameras();

    if (virtualCameras.empty()) return;

    auto updateTimeAndCheckIfDead = [elapsedTimeMs](VirtualCamera& vc) {
        vc.timeToLiveMs -= elapsedTimeMs;
        return vc.timeToLiveMs <= 0;
    };

    // Remove virtual cameras with time to live smaller or equal to zero
    removeElementsAndPack(virtualCameras, updateTimeAndCheckIfDead);

    // Calculate the ratio to update the angles and angle spans
    float updateRatio = std::min(float(elapsedTimeMs) / TIME_TO_GOAL_MS, 1.f);

    // Move the virtual cameras toward their goal position
    for (VirtualCamera& vc : virtualCameras)
    {
        float azimuthDifference = math::getSignedAzimuthDifference(vc.azimuth, vc.goal.azimuth);
        float elevationDifference = vc.goal.elevation - vc.elevation;
        float distance =
            math::getApproximatedSphericalAnglesDistance(vc.elevation, azimuthDifference, elevationDifference);

        // Update position if required
        if (distance > POSITION_CHANGED_THRESHOLD)
        {
            vc.azimuth = std::fmod(vc.azimuth + azimuthDifference * updateRatio, 2 * math::PI);
            vc.elevation += elevationDifference * updateRatio;
        }

        // Update size if required
        if (std::abs(vc.goal.azimuthSpan - vc.azimuthSpan) > DIMENTION_CHANGED_THRESHOLD &&
            std::abs(vc.goal.elevationSpan - vc.elevationSpan) > DIMENTION_CHANGED_THRESHOLD)
        {
            float resizeFactor = 1 + (((vc.goal.elevationSpan / vc.elevationSpan) - 1) * updateRatio);
            vc.elevationSpan *= resizeFactor;
            vc.azimuthSpan *= resizeFactor;
        }
    }

    setVirtualCameras(virtualCameras);
}

void VirtualCameraManager::updateVirtualCamerasGoal(const std::vector<SphericalAngleRect>& goals)
{
    if (goals.empty()) return;

    // Get a vector of non duplicate goals (duplicates are within a set threshold)
    std::vector<SphericalAngleRect> uniqueGoals = getUniqueRegions(goals);

    // Update the goals elevation and spans to be within bounds
    updateRegionsBounds(uniqueGoals);

    // Get the distance between each virtual camera and each goal in a sorted map (map values are indices pair)
    const std::multimap<float, std::pair<int, int>> orderedDistanceMap = getVcToGoalOrderedDistances(uniqueGoals);

    // Create vectors to keep track of which virtual camera and goal were matched
    const int vcCount = virtualCameras_.size();
    const int goalCount = uniqueGoals.size();
    std::vector<bool> isVcUnmatchedVec(vcCount, true);
    std::vector<bool> isGoalUnmatchedVec(goalCount, true);

    int matchedVcCount = 0;
    int matchedGoalCount = 0;

    // Update the virtual camera goals
    for (const auto& entry : orderedDistanceMap)
    {
        int vcIndex = entry.second.first;
        int goalIndex = entry.second.second;

        // Make sure the virtual camera and the goal are unmatched
        if (isVcUnmatchedVec[vcIndex] && isGoalUnmatchedVec[goalIndex])
        {
            isVcUnmatchedVec[vcIndex] = false;
            isGoalUnmatchedVec[goalIndex] = false;
            ++matchedVcCount;
            ++matchedGoalCount;

            // Update the virtual camera goal and reset its time to live
            virtualCameras_[vcIndex].goal = uniqueGoals[goalIndex];
            virtualCameras_[vcIndex].timeToLiveMs = VIRTUAL_CAMERA_TIME_TO_LIVE_MS;

            // If all virtual cameras or all goals were matched, no reason to keep going
            if (matchedVcCount == vcCount || matchedGoalCount == goalCount)
            {
                break;
            }
        }
    }

    // Create new virtual cameras with unmatched goals
    if (matchedGoalCount != goalCount)
    {
        for (int i = 0; i < goalCount; ++i)
        {
            if (isGoalUnmatchedVec[i])
            {
                virtualCameras_.emplace_back(uniqueGoals[i], VIRTUAL_CAMERA_TIME_TO_LIVE_MS);
            }
        }
    }
}

void VirtualCameraManager::clearVirtualCameras()
{
    virtualCameras_.clear();
}

std::vector<VirtualCamera> VirtualCameraManager::getVirtualCameras()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return virtualCameras_;
}

 void VirtualCameraManager::setVirtualCameras(const std::vector<VirtualCamera>& virtualCameras)
 {
     std::lock_guard<std::mutex> lock(mutex_);
     virtualCameras_ = virtualCameras;
 }

float VirtualCameraManager::getElevationOverflow(float elevation, float elevationSpan)
{
    float elevationOverflow = 0.f;
    float halfElevationSpan = elevationSpan / 2.f;

    if (elevation - halfElevationSpan < srcImageMinElevation_)
    {
        elevationOverflow = elevation - halfElevationSpan - srcImageMinElevation_;
    }
    else if (elevation + halfElevationSpan > srcImageMaxElevation_)
    {
        elevationOverflow = elevation + halfElevationSpan - srcImageMaxElevation_;
    }

    return elevationOverflow;
}

std::vector<SphericalAngleRect> VirtualCameraManager::getUniqueRegions(const std::vector<SphericalAngleRect>& regions)
{
    std::vector<SphericalAngleRect> uniqueRegions;
    uniqueRegions.reserve(regions.size());

    for (const SphericalAngleRect& region : regions)
    {
        bool isGoalDuplicate = false;

        // Check if there is already a region within the angle thresholds
        for (const SphericalAngleRect& uniqueRegion : uniqueRegions)
        {
            if (region.azimuth > uniqueRegion.azimuth - DUPLICATE_FACE_ANGLE_RANGE &&
                region.azimuth < uniqueRegion.azimuth + DUPLICATE_FACE_ANGLE_RANGE &&
                region.elevation > uniqueRegion.elevation - DUPLICATE_FACE_ANGLE_RANGE &&
                region.elevation < uniqueRegion.elevation + DUPLICATE_FACE_ANGLE_RANGE)
            {
                isGoalDuplicate = true;
                break;
            }
        }

        // If this region is not a duplicate, add it to the unique regions
        if (!isGoalDuplicate)
        {
            uniqueRegions.emplace_back(region.azimuth, region.elevation, region.azimuthSpan, region.elevationSpan);
        }
    }

    return uniqueRegions;
}

void VirtualCameraManager::updateRegionsBounds(std::vector<SphericalAngleRect>& regions)
{
    for (SphericalAngleRect& region : regions)
    {
        float newElevation = region.elevation + region.elevationSpan / ELEVATION_SHIFT_RATIO;
        region.elevationSpan = 
            math::clamp(region.elevationSpan, VIRTUAL_CAMERA_MIN_ELEVATION_SPAN, srcImageMaxElevationSpan_);
        region.azimuthSpan = region.elevationSpan * aspectRatio_;
        region.elevation = newElevation - getElevationOverflow(newElevation, region.elevationSpan);
    }
}

std::multimap<float, std::pair<int, int>> VirtualCameraManager::getVcToGoalOrderedDistances(
    const std::vector<SphericalAngleRect>& goals)
{
    std::multimap<float, std::pair<int, int>> distanceIndicesMap;
    const int regionCount = virtualCameras_.size();
    const int goalsCount = goals.size();

    for (int regionIndex = 0; regionIndex < regionCount; ++regionIndex)
    {
        const SphericalAngleRect& region = virtualCameras_[regionIndex];

        for (int goalIndex = 0; goalIndex < goalsCount; ++goalIndex)
        {
            const SphericalAngleRect& goal = goals[goalIndex];
            float distance = math::getApproximatedSphericalAnglesDistance(region.azimuth, region.elevation,
                                                                          goal.azimuth, goal.elevation);
            distanceIndicesMap.emplace(distance, std::pair<int, int>(regionIndex, goalIndex));
        }
    }

    return distanceIndicesMap;
}

}    // namespace Model
