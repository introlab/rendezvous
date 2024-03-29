#include "image_file_reader.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

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
            RGBImage rgbImage(width, height);
            rgbImage.hostData = imageData;
            image_ = Image(width, height, format);
            heapObjectFactory_.allocateObject(image_);
            imageConverter_.convert(rgbImage, image_);
        }
        else
        {
            image_ = RGBImage(width, height);
            image_.hostData = imageData;
        }
    }
    else
    {
        image_.hostData = nullptr;
        throw std::runtime_error("Could not open file " + imageFilePath);
    }
}

void ImageFileReader::open()
{
    // Nothing to be done
    // TODO: breaks interfaces segregation principle
}

void ImageFileReader::close()
{
    // Nothing to be done
    // TODO: breaks interfaces segregation principle
}

bool ImageFileReader::readImage(Image& image)
{
    image = image_;
    return true;
}

bool ImageFileReader::loadImage(const char* fileName, unsigned char*& data, int& width, int& height, int& channels,
                                int desiredChannels) const
{
    data = stbi_load(fileName, &width, &height, &channels, desiredChannels);

    if (desiredChannels != 0)
    {
        channels = desiredChannels;
    }

    return data != nullptr;
}

}    // namespace Model
