#ifndef IMAGE_FILE_READER_H
#define IMAGE_FILE_READER_H

#include <memory>
#include <string>

#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_converter.h"
#include "model/stream/utils/images/image_format.h"
#include "model/stream/video/input/i_video_input.h"

namespace Model
{
class ImageFileReader : public IVideoInput
{
   public:
    ImageFileReader(const std::string& imageFilePath, ImageFormat format);

    void open() override;
    void close() override;
    const Image& readImage() override;

   protected:
    Image image_;

   private:
    bool loadImage(const char* fileName, unsigned char*& data, int& width, int& height, int& channels,
                   int desiredChannels = 0) const;

    ImageConverter imageConverter_;
    HeapObjectFactory heapObjectFactory_;
};

}    // namespace Model

#endif    // !IMAGE_FILE_READER_H
