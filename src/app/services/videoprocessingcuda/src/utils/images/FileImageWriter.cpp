#include "FileImageWriter.h"

#include "utils/images/stb/stb_image_write.h"

FileImageWriter::FileImageWriter(const std::string& folder, const std::string& imagesName)
    : folder_(folder)
    , imagesName_(imagesName)
    , imageNameIndex_(0)
{
}

void FileImageWriter::consumeImage(const Image& image)
{
    stbi_write_png((folder_ + "/" + imagesName_ + std::to_string(imageNameIndex_++) + ".png").c_str(), image.width,
                   image.height, image.channels, image.hostData, 0);
}