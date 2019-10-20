#include "VideoStreamMock.h"

#include <stdexcept>
#include <iostream>
#include <cstring>

VideoStreamMock::VideoStreamMock(const std::string& imageFilePath)
{
    stbi_uc* imageData;
    int width, height, channels;

    if (loadImage(imageFilePath.c_str(), imageData, width, height, channels, STBI_rgb))
    {
        image_ = Image(width , height , channels);
        image_.hostData = imageData;
    }
    else
    {
        image_.hostData = nullptr;
        throw std::runtime_error("Could not open file " + imageFilePath);
    }
}

bool VideoStreamMock::copyFrameData(const Image& image)
{
    if (image_.hostData && image.size == image_.size)
    {
        std::memcpy(image.hostData, image_.hostData, image_.size * sizeof(unsigned char));
    }

    return image_.hostData != nullptr;
}

bool VideoStreamMock::getResolution(Dim3<int>& dim)
{
    if (image_.hostData)
    {
        dim.width = image_.width;
        dim.height = image_.height;
        dim.channels = image_.channels;
    }

    return image_.hostData != nullptr;
}

bool VideoStreamMock::loadImage(const char* fileName, stbi_uc*& data, int& width, int& height, int& channels, int desiredChannels) const
{
    data = stbi_load(fileName, &width, &height, &channels, desiredChannels);

    if (desiredChannels != 0)
    {
        channels = desiredChannels;
    }

    return data != nullptr;
}
