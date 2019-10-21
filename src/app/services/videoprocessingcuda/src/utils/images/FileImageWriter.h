#ifndef FILE_IMAGE_WRITER_H
#define FILE_IMAGE_WRITER_H

#include <string>

#include "IImageConsumer.h"

class FileImageWriter : public IImageConsumer
{
public:

    FileImageWriter(const std::string& folder, const std::string& imagesName);

    void consumeImage(const Image& image) override;

private:

    std::string folder_;
    std::string imagesName_;
    int imageNameIndex_;

};

#endif // !FILE_IMAGE_WRITER_H
