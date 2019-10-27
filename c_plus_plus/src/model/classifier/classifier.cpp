#include "classifier.h"

namespace Model
{

std::vector<int> Classifier::classify(std::vector<SourcePosition> audioPositions, std::vector<SourcePosition> imagePositions, int rangeThreshold)
{
    std::vector<int> sources;

    for(size_t i = 0; i < audioPositions.size(); i++)
    {
        if(audioPositions[i].elevation > 0)
        {
            for(size_t j = 0; j < imagePositions.size(); j++)
            {
                if(std::abs(audioPositions[i].azimuth - imagePositions[j].azimuth) <= rangeThreshold
                        && std::abs(audioPositions[i].elevation - imagePositions[j].elevation) <= rangeThreshold)
                {
                    sources.push_back(static_cast<int>(i));
                }
            }
        }
    }

    return sources;
}

}
