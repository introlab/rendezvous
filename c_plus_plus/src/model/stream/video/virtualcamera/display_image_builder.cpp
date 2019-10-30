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
const RGB RGB_BACKGROUND = {220, 220, 220};
const UYVY UYVY_BACKGROUND = {128, 220, 128, 220};
const YUYV YUYV_BACKGROUND = {220, 128, 220, 128};

template <typename T>
void setColor(T* data, int size, T color)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = color;
    }
}

template <typename T>
void fillImage(int offset, const Dim2<int>& inputDim, const Dim2<int>& outputDim, const T* inputData, T* outputData)
{
    for (int i = 0; i < inputDim.width; ++i)
    {
        for (int j = 0; j < inputDim.height; ++j)
        {
            outputData[offset + i + j * outputDim.width] = inputData[i + j * inputDim.width];
        }
    }
}

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

const Dim2<int>& DisplayImageBuilder::getMaxVirtualCameraDim()
{
    return maxVirtualCameraDim_;
}

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
                const RGB* inputData = reinterpret_cast<const RGB*>(vcImage.hostData);
                RGB* outputData = reinterpret_cast<RGB*>(outDisplayImage.hostData);
                fillImage(offset, vcImage, outDisplayImage, inputData, outputData);
            }
            else if (vcImage.format == ImageFormat::UYVY_FMT && outDisplayImage.format == ImageFormat::UYVY_FMT)
            {
                const UYVY* inputData = reinterpret_cast<const UYVY*>(vcImage.hostData);
                UYVY* outputData = reinterpret_cast<UYVY*>(outDisplayImage.hostData);
                Dim2<int> inputDim(vcImage.width / 2, vcImage.height);
                Dim2<int> outputDim(outDisplayImage.width / 2, outDisplayImage.height);
                fillImage(offset / 2, inputDim, outputDim, inputData, outputData);
            }
            else if (vcImage.format == ImageFormat::YUYV_FMT && outDisplayImage.format == ImageFormat::YUYV_FMT)
            {
                const UYVY* inputData = reinterpret_cast<const UYVY*>(vcImage.hostData);
                UYVY* outputData = reinterpret_cast<UYVY*>(outDisplayImage.hostData);
                Dim2<int> inputDim(vcImage.width / 2, vcImage.height);
                Dim2<int> outputDim(outDisplayImage.width / 2, outDisplayImage.height);
                fillImage(offset / 2, inputDim, outputDim, inputData, outputData);
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
    int size = displayImage.width * displayImage.height;

    if (displayImage.format == ImageFormat::RGB_FMT)
    {
        RGB* data = reinterpret_cast<RGB*>(displayImage.hostData);
        setColor(data, size, RGB_BACKGROUND);
    }
    else if (displayImage.format == ImageFormat::UYVY_FMT)
    {
        UYVY* data = reinterpret_cast<UYVY*>(displayImage.hostData);
        setColor(data, size / 2, UYVY_BACKGROUND);
    }
    else if (displayImage.format == ImageFormat::YUYV_FMT)
    {
        YUYV* data = reinterpret_cast<YUYV*>(displayImage.hostData);
        setColor(data, size / 2, YUYV_BACKGROUND);
    }
}

void DisplayImageBuilder::clearVirtualCamerasOnDisplayImage(const Image& displayImage)
{
    int memsetSize = maxVirtualCameraDim_.height * (displayImage.width * displayImage.bytesPerPixel);
    int memsetOffset =
        ((displayImage.height - maxVirtualCameraDim_.height) / 2) * (displayImage.width * displayImage.bytesPerPixel);
    std::memset(displayImage.hostData + memsetOffset, BACKGROUND_VALUE, memsetSize);
}
}    // namespace Model
