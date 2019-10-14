#ifndef I_DETECTOR_MOCK_H
#define I_DETECTOR_MOCK_H

#include "IDetector.h"

class DetectorMock : public IDetector
{
public:

    std::vector<Rectangle> detectInImage(const ImageFloat& image) override;
    Dim2<int> getInputImageDim() override;

};

#endif //!I_DETECTOR_MOCK_H