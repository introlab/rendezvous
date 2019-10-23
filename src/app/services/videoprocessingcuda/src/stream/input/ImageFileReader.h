#ifndef IMAGE_FILE_READER_H
#define IMAGE_FILE_READER_H

#include <string>
#include <memory>

#include "stream/input/IVideoInput.h"
#include "utils/images/ImageFormat.h"
#include "utils/images/ImageConverter.h"
#include "utils/alloc/HeapObjectFactory.h"

class ImageFileReader : public IVideoInput
{
public:

    ImageFileReader(const std::string& imageFilePath, ImageFormat format);

    const Image& readImage() override;

protected:

    Image image_;

private:

    bool loadImage(const char* fileName, unsigned char*& data, int& width, int& height, int& channels, int desiredChannels = 0) const;

    ImageConverter imageConverter_;
    HeapObjectFactory heapObjectFactory_;

};

#endif // !IMAGE_FILE_READER_H
