#ifndef DETECTOR_MOCK_H
#define DETECTOR_MOCK_H

#include "IDetector.h"

class DetectorMock : public IDetector
{
public:

    std::vector<Rectangle> detectInImage(const ImageFloat& image) override;
    Dim2<int> getInputImageDim() override;

};

#endif //!DETECTOR_MOCK_H