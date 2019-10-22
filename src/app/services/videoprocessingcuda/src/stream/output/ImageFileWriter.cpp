#include "ImageFileWriter.h"

#include "utils/images/stb/stb_image_write.h"

ImageFileWriter::ImageFileWriter(const std::string& folder, const std::string& imageName)
    : folder_(folder)
    , imageName_(imageName)
    , image_(RGBImage(0,0))
{
}

ImageFileWriter::~ImageFileWriter()
{
    objectFactory_.deallocateObject(image_);
}

void ImageFileWriter::writeImage(const Image& image)
{
    Image outputImage;
    
    if (image.format != ImageFormat::RGB_FMT)
    {
        if (image_.width * image_.height < image.width * image.height)
        {
            objectFactory_.deallocateObject(image_);
            image_ = RGBImage(image.width, image.height);
            objectFactory_.allocateObject(image_);
        }

        imageConverter_.convert(image, image_);
        outputImage = image_;
    }
    else
    {
        outputImage = image;
    }
    
    stbi_write_png((folder_ + "/" + imageName_ + ".png").c_str(), outputImage.width, outputImage.height, 3, outputImage.hostData, 0);
}