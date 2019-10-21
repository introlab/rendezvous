#include "DisplayImageBuilder.h"

#include <cstring>

namespace
{
    const int VIRTUAL_CAMERA_SPACING = 10;
    const int DISPLAY_HEIGHT_SPACING = 50;
    const int RGB_BACKGROUND_VALUE = 222;
}

DisplayImageBuilder::DisplayImageBuilder(const Dim2<int>& displayDimention)
    : displayDimention_(displayDimention)
{
    int maxVcSize = displayDimention_.height - DISPLAY_HEIGHT_SPACING;
    maxVirtualCameraDim_ = Dim2<int>(maxVcSize, maxVcSize); // For now virtual camera need to be square
}

Dim2<int> DisplayImageBuilder::getVirtualCameraDim(int virtualCameraCount)
{
    int vcWidth = 0;
    int vcHeight = 0;
    
    if (virtualCameraCount > 0)
    {
        int availableWidth = displayDimention_.width - (VIRTUAL_CAMERA_SPACING * (virtualCameraCount + 1));
        vcWidth = availableWidth / virtualCameraCount;
        vcHeight = displayDimention_.height - DISPLAY_HEIGHT_SPACING;

        if (vcWidth < vcHeight)
        {
            vcHeight = vcWidth;
        }
        else
        {
            vcWidth = vcHeight;
        }
    }
    
    return Dim2<int>(vcWidth, vcHeight);
}

Dim2<int> DisplayImageBuilder::getMaxVirtualCameraDim()
{
    return maxVirtualCameraDim_;
}

void DisplayImageBuilder::createDisplayImage(const std::vector<Image>& vcImages, const Image& outDisplayImage)
{
    int vcCount = (int)vcImages.size();
    if (displayDimention_ == outDisplayImage && vcCount > 0)
    {
        Dim3<int> vcDim = vcImages[0];
        int firstVcLeftOffset = ((outDisplayImage.width - vcCount * vcDim.width - (vcCount - 1) * VIRTUAL_CAMERA_SPACING) / 2) * vcDim.channels;
        int topOffset = ((outDisplayImage.height - vcDim.height) / 2) * (outDisplayImage.width * vcDim.channels);

        for (int k = 0; k < vcCount; ++k)
        {
            const Image& vcImage = vcImages[k];
            int leftOffset = firstVcLeftOffset + k * (vcDim.width + VIRTUAL_CAMERA_SPACING) * vcDim.channels;
            int offset = topOffset + leftOffset;

            for (int i = 0; i < vcDim.width * 3; ++i)
            {
                for (int j = 0; j < vcDim.height; ++j)
                {
                    outDisplayImage.hostData[offset + i + j * (outDisplayImage.width * vcDim.channels)] = vcImage.hostData[i + j * (vcDim.width * vcDim.channels)];
                }
            }
        }
    }
}

void DisplayImageBuilder::setDisplayImageColor(const Image& displayImage)
{
     std::memset(displayImage.hostData , RGB_BACKGROUND_VALUE, displayImage.size);
}

void DisplayImageBuilder::clearVirtualCamerasOnDisplayImage(const Image& displayImage)
{
    int memsetSize = maxVirtualCameraDim_.height * (displayImage.width * displayImage.channels);
    int memsetOffset = ((displayImage.height - maxVirtualCameraDim_.height) / 2) * (displayImage.width * displayImage.channels);
    std::memset(displayImage.hostData + memsetOffset, RGB_BACKGROUND_VALUE, memsetSize);
}