#ifndef DISPLAY_IMAGE_BUILDER_H
#define DISPLAY_IMAGE_BUILDER_H

#include <vector>

#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/dim2.h"

namespace Model
{
class DisplayImageBuilder
{
   public:
    explicit DisplayImageBuilder(const Dim2<int>& displayDimention);

    Dim2<int> getVirtualCameraDim(int virtualCameraCount);
    const Dim2<int>& getMaxVirtualCameraDim();
    void createDisplayImage(const std::vector<Image>& vcImages, const Image& outDisplayImage);
    void setDisplayImageColor(const Image& displayImage);
    void clearVirtualCamerasOnDisplayImage(const Image& displayImage);

   private:
    Dim2<int> displayDimention_;
    Dim2<int> maxVirtualCameraDim_;
};

}    // namespace Model

#endif    //! DISPLAY_IMAGE_BUILDER_H
