#ifndef DISPLAY_IMAGE_BUILDER_H
#define DISPLAY_IMAGE_BUILDER_H

#include <vector>

#include "utils/models/Dim2.h"
#include "utils/images/Images.h"

class DisplayImageBuilder
{
public:

    DisplayImageBuilder(const Dim2<int>& displayDimention);

    Dim2<int> getVirtualCameraDim(int virtualCameraCount);
    Dim2<int> getMaxVirtualCameraDim();
    void createDisplayImage(const std::vector<Image>& vcImages, const Image& outDisplayImage);
    void setDisplayImageColor(const Image& displayImage);
    void clearVirtualCamerasOnDisplayImage(const Image& displayImage);

private:

    Dim2<int> displayDimention_;
    Dim2<int> maxVirtualCameraDim_;

};

#endif //!DISPLAY_IMAGE_BUILDER_H