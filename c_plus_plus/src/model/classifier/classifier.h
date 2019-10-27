#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>

#include "model/audio/source_position.h"

namespace Model
{

class Classifier
{
public:
    // TODO: change the return type if its not the index that we want
    // TODO: change SourcePosition for ImagePosition
    static std::vector<int> classify(std::vector<SourcePosition> audioPositions, std::vector<SourcePosition> imagePositions, int rangeThreshold);
};

}   // Model

#endif // CLASSIFIER_H
