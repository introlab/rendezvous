#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>

#include "model/stream/audio/source_position.h"
#include "model/stream/utils/models/spherical_angle_rect.h"

namespace Model
{

class Classifier
{
public:
    static std::vector<int> classify(std::vector<SourcePosition> audioPositions, std::vector<SphericalAngleRect> imagePositions, int rangeThreshold);
};

}   // Model

#endif // CLASSIFIER_H
