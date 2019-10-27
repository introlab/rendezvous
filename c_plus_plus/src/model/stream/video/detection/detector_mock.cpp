#include "detector_mock.h"

#include <random>

namespace Model
{
std::vector<Rectangle> DetectorMock::detectInImage(const ImageFloat& image)
{
    float width = 400;
    float height = 300;

    std::uniform_int_distribution<std::mt19937::result_type> xDistribution(0, static_cast<int>(image.width - width));
    std::uniform_int_distribution<std::mt19937::result_type> yDistribution(0, static_cast<int>(image.height - height));

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());

    float x = static_cast<float>(xDistribution(generator)) + (width / 2);
    float y = static_cast<float>(yDistribution(generator)) + (height / 2);

    return std::vector<Rectangle>{Rectangle(x, y, width, height)};
}

Dim2<int> DetectorMock::getInputImageDim() { return Dim2<int>(800, 600); }

}    // namespace Model
