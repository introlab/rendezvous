#include "image_file_reader.h"

#include <stdexcept>
#include <iostream>
#include <cstring>

#include "model/stream/utils/images/stb/stb_image.h"

namespace Model
{

ImageFileReader::ImageFileReader(const std::string& imageFilePath, ImageFormat format)
{
    stbi_uc* imageData;
    int width, height, channels;

    if (loadImage(imageFilePath.c_str(), imageData, width, height, channels, STBI_rgb))
    {
        if (format != ImageFormat::RGB_FMT)
        {
            RGBImage rgbImage(width , height);
            rgbImage.hostData = imageData;
            image_ = Image(width , height, format);
            heapObjectFactory_.allocateObject(image_);
            imageConverter_.convert(rgbImage, image_);
        }
        else
        {
            image_ = RGBImage(width , height);
            image_.hostData = imageData;
        }
    }
    else
    {
        image_.hostData = nullptr;
        throw std::runtime_error("Could not open file " + imageFilePath);
    }
}

const Image& ImageFileReader::readImage()
{
    return image_;
}

bool ImageFileReader::loadImage(const char* fileName, unsigned char*& data, int& width, int& height, int& channels, int desiredChannels) const
{
    data = stbi_load(fileName, &width, &height, &channels, desiredChannels);

    if (desiredChannels != 0)
    {
        channels = desiredChannels;
    }

    return data != nullptr;
}

} // Model
