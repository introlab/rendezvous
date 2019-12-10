#ifndef DETECTION_DEWARP_OPTIMIZER_H
#define DETECTION_DEWARP_OPTIMIZER_H

#include <vector>
#include <deque>

namespace Model
{
class DetectionDewarpingOptimizer
{
   public:
    DetectionDewarpingOptimizer(int dewarpAreaCount, int planLength);

    void incrementDetectionInArea(int areaIndex);
    std::vector<int> getNextDetectionAreas();

   private:
    int dewarpAreaCount_;
    int planLength_;

    std::vector<int> detectionInAreaOccurences_;
    std::vector<int> maxDetectionInAreaPerPlan_;
    std::deque<std::vector<int>> detectionAreaPlan_;
};
}   // Model

#endif  // DETECTION_DEWARP_OPTIMIZER_H