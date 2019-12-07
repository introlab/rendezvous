#include "detection_dewarp_optimizer.h"

namespace Model
{

DetectionDewarpingOptimizer::DetectionDewarpingOptimizer(int dewarpAreaCount, int planLength)
    : dewarpAreaCount_(dewarpAreaCount)
    , planLength_(planLength)
    , detectionInAreaOccurences_(dewarpAreaCount, 0)
    , maxDetectionInAreaPerPlan_(dewarpAreaCount, planLength)
    , detectionAreaPlan_(planLength)
{
    std::vector<int> nextDetectionAreas;
    
    for (int i = 0; i < dewarpAreaCount; ++i)
    {
        nextDetectionAreas.push_back(i);
    }

    for (int i = 0; i < planLength; ++i)
    {
        detectionAreaPlan_[i] = nextDetectionAreas;
    }
}

void DetectionDewarpingOptimizer::incrementDetectionInArea(int areaIndex)
{
    ++detectionInAreaOccurences_[areaIndex];
}

std::vector<int> DetectionDewarpingOptimizer::getNextDetectionAreas()
{
    if (detectionAreaPlan_.empty())
    {
        detectionAreaPlan_.assign(planLength_, {});

        for (int areaIndex = 0; areaIndex < dewarpAreaCount_; ++areaIndex)
        {
            float detectionRatio = static_cast<float>(detectionInAreaOccurences_[areaIndex]) / 
                                   static_cast<float>(maxDetectionInAreaPerPlan_[areaIndex]);
            
            maxDetectionInAreaPerPlan_[areaIndex] = 0;
            detectionInAreaOccurences_[areaIndex] = 0;

            int nextDewarpingCount = std::max(1, std::min(static_cast<int>(detectionRatio * planLength_), planLength_));
            
            for (int planIndex = 0; planIndex < nextDewarpingCount; ++planIndex)
            {
                detectionAreaPlan_[planIndex].push_back(areaIndex);
                ++maxDetectionInAreaPerPlan_[areaIndex];
            }
        }
    }

    std::vector<int> nextDetectionAreas = detectionAreaPlan_.front();
    detectionAreaPlan_.pop_front();

    return nextDetectionAreas;
}

}   // Model