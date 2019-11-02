#ifndef DETECTOR_MOCK_H
#define DETECTOR_MOCK_H

#include "model/stream/video/detection/i_detector.h"

namespace Model
{
class DetectorMock : public IDetector
{
   public:
    std::vector<Rectangle> detectInImage(const ImageFloat& image) override;
    Dim2<int> getInputImageDim() override;
};

}    // namespace Model

#endif    //! DETECTOR_MOCK_H
