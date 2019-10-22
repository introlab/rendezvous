#ifndef VIDEO_STREAM_MOCK_H
#define VIDEO_STREAM_MOCK_H

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

private:

    bool loadImage(const char* fileName, unsigned char*& data, int& width, int& height, int& channels, int desiredChannels = 0) const;

    Image image_;
    ImageConverter imageConverter_;
    HeapObjectFactory heapObjectFactory_;

};

#endif // !VIDEO_STREAM_MOCK_H
