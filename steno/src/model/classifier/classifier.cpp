#include <cmath>

#include "classifier.h"

namespace Model
{
/**
 * @brief Uses the audio/image pairs to determine which source we want to keep.
 * @param audioPositions - audio positions from odas localization.
 * @param imagePositions - image positions to compare with audio positions.
 * @param rangeThreshold - distance threshold
 * @return vector of index of the sources we want to keep in the audio.
 */
std::vector<int> Classifier::getSourcesToKeep(const std::vector<SourcePosition> &audioPositions,
                                              const std::vector<SphericalAngleRect> &imagePositions,
                                              const float &rangeThreshold)
{
    std::vector<int> sourcesToKeep;
    std::vector<std::pair<int, int>> audioImagePairs =
        getAudioImagePairs(audioPositions, imagePositions, rangeThreshold);

    for (size_t i = 0; i < audioImagePairs.size(); i++)
    {
        std::pair<int, int> pair = audioImagePairs[i];

        // Check if the index to keep is not already in the vector
        if (std::find(sourcesToKeep.begin(), sourcesToKeep.end(), pair.first) == sourcesToKeep.end())
        {
            sourcesToKeep.push_back(pair.first);
        }
    }

    return sourcesToKeep;
}

/**
 * @brief Compares the spatial positions of audio sources and images to identify audio/image pairs
 * @param audioPositions - audio positions from odas localization.
 * @param imagePositions - image positions to compare with audio positions.
 * @param rangeThreshold - distance threshold
 * @return vector of pairs of index that represents the audio sources and images that are at the same spatial position
 */
std::vector<std::pair<int, int>> Classifier::getAudioImagePairs(const std::vector<SourcePosition> &audioPositions,
                                                                const std::vector<SphericalAngleRect> &imagePositions,
                                                                const float &rangeThreshold)
{
    std::pair<int, int> audioImagePair;
    std::vector<std::pair<int, int>> audioImagePairs;

    for (size_t i = 0; i < audioPositions.size(); i++)
    {
        for (size_t j = 0; j < imagePositions.size(); j++)
        {
            // Set the azimuth counter-clockwise to match the audio
            float imagePositionAzimuth = 2.0 * M_PI - imagePositions[j].azimuth;

            if (std::abs(audioPositions[i].azimuth - imagePositionAzimuth) < rangeThreshold &&
                std::abs(audioPositions[i].elevation - imagePositions[j].elevation) < rangeThreshold)
            {
                audioImagePairs.push_back(std::make_pair(i, j));
            }
        }
    }

    return audioImagePairs;
}

}    // namespace Model
