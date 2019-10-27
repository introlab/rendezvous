#include "display_image_builder.h"

#include <cstring>
#include <stdexcept>

namespace Model
{
namespace
{
const int VIRTUAL_CAMERA_SPACING = 10;
const int DISPLAY_HEIGHT_SPACING = 50;
const int BACKGROUND_VALUE = 128;
}    // namespace

DisplayImageBuilder::DisplayImageBuilder(const Dim2<int>& displayDimention)
    : displayDimention_(displayDimention)
{
    int maxVcSize = displayDimention_.height - DISPLAY_HEIGHT_SPACING;
    maxVirtualCameraDim_ = Dim2<int>(maxVcSize, maxVcSize);    // For now virtual camera need to be square
}

Dim2<int> DisplayImageBuilder::getVirtualCameraDim(int virtualCameraCount)
{
    int vcWidth = 0;
    int vcHeight = 0;

    if (virtualCameraCount > 0)
    {
        int availableWidth = displayDimention_.width - (VIRTUAL_CAMERA_SPACING * (virtualCameraCount + 1));

        // Make sure it's a multiple of 2 for compressed formats (make last bit 0)
        vcWidth = (availableWidth / virtualCameraCount) & 0xFFFE;
        vcHeight = (displayDimention_.height - DISPLAY_HEIGHT_SPACING) & 0xFFFE;

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

Dim2<int> DisplayImageBuilder::getMaxVirtualCameraDim() { return maxVirtualCameraDim_; }

void DisplayImageBuilder::createDisplayImage(const std::vector<Image>& vcImages, const Image& outDisplayImage)
{
    int vcCount = (int)vcImages.size();
    if (displayDimention_ == outDisplayImage && vcCount > 0)
    {
        Dim2<int> vcDim = vcImages[0];
        int firstVcLeftOffset =
            (outDisplayImage.width - vcCount * vcDim.width - (vcCount - 1) * VIRTUAL_CAMERA_SPACING) / 2;
        int topOffset = ((outDisplayImage.height - vcDim.height) / 2) * outDisplayImage.width;

        for (int vcIndex = 0; vcIndex < vcCount; ++vcIndex)
        {
            const Image& vcImage = vcImages[vcIndex];
            int leftOffset = firstVcLeftOffset + vcIndex * (vcDim.width + VIRTUAL_CAMERA_SPACING);
            int offset = topOffset + leftOffset;

            if (vcImage.format == ImageFormat::RGB_FMT && outDisplayImage.format == ImageFormat::RGB_FMT)
            {
                const RGB* vcData = reinterpret_cast<const RGB*>(vcImage.hostData);
                RGB* displayData = reinterpret_cast<RGB*>(outDisplayImage.hostData);
                fillImage(offset, vcImage, outDisplayImage, vcData, displayData);
            }
            else if (vcImage.format == ImageFormat::UYVY_FMT && outDisplayImage.format == ImageFormat::UYVY_FMT)
            {
                const UYVY* vcData = reinterpret_cast<const UYVY*>(vcImage.hostData);
                UYVY* displayData = reinterpret_cast<UYVY*>(outDisplayImage.hostData);
                fillImage(offset, vcImage, outDisplayImage, vcData, displayData);
            }
            else if (vcImage.format == ImageFormat::YUYV_FMT && outDisplayImage.format == ImageFormat::YUYV_FMT)
            {
                const UYVY* vcData = reinterpret_cast<const UYVY*>(vcImage.hostData);
                UYVY* displayData = reinterpret_cast<UYVY*>(outDisplayImage.hostData);
                fillImage(offset, vcImage, outDisplayImage, vcData, displayData);
            }
            else
            {
                throw std::invalid_argument("Display image builder error, formats are not the same : VC (" +
                                            getImageFormatString(vcImage.format) + ") and Display (" +
                                            getImageFormatString(outDisplayImage.format) + ")");
            }
        }
    }
}

void DisplayImageBuilder::setDisplayImageColor(const Image& displayImage)
{
    // This doesn't provide the color we want, but is much more efficient (darker gray)
    // When we modify the code to dewarp directly into the display image, we'll fix the image color
    std::memset(displayImage.hostData, BACKGROUND_VALUE, displayImage.size);
}

void DisplayImageBuilder::clearVirtualCamerasOnDisplayImage(const Image& displayImage)
{
    int memsetSize = maxVirtualCameraDim_.height * (displayImage.width * displayImage.bytesPerPixel);
    int memsetOffset =
        ((displayImage.height - maxVirtualCameraDim_.height) / 2) * (displayImage.width * displayImage.bytesPerPixel);
    std::memset(displayImage.hostData + memsetOffset, BACKGROUND_VALUE, memsetSize);
}

void DisplayImageBuilder::fillImage(int offset, const Dim2<int>& vcDim, const Dim2<int>& displayDim, const RGB* vcData,
                                    RGB* displayData)
{
    for (int i = 0; i < vcDim.width; ++i)
    {
        for (int j = 0; j < vcDim.height; ++j)
        {
            displayData[offset + i + j * displayDim.width] = vcData[i + j * vcDim.width];
        }
    }
}

void DisplayImageBuilder::fillImage(int offset, Dim2<int> vcDim, Dim2<int> displayDim, const UYVY* vcData,
                                    UYVY* displayData)
{
    // Each UYVY struct is 2 pixels so we need to divide by 2
    offset /= 2;
    vcDim.width /= 2;
    displayDim.width /= 2;

    for (int i = 0; i < vcDim.width; ++i)
    {
        for (int j = 0; j < vcDim.height; ++j)
        {
            displayData[offset + i + j * displayDim.width] = vcData[i + j * vcDim.width];
        }
    }
}

void DisplayImageBuilder::fillImage(int offset, Dim2<int> vcDim, Dim2<int> displayDim, const YUYV* vcData,
                                    YUYV* displayData)
{
    // Each YUYV struct is 2 pixels so we need to divide by 2
    offset /= 2;
    vcDim.width /= 2;
    displayDim.width /= 2;

    for (int i = 0; i < vcDim.width; ++i)
    {
        for (int j = 0; j < vcDim.height; ++j)
        {
            displayData[offset + i + j * displayDim.width] = vcData[i + j * vcDim.width];
        }
    }
}
}    // namespace Model
