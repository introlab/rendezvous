#ifndef IMAGE_FILE_WRITER_H
#define IMAGE_FILE_WRITER_H

#include <memory>

#include "model/stream/utils/alloc/heap_object_factory.h"
#include "model/stream/utils/images/image_converter.h"
#include "model/stream/video/output/i_video_output.h"

namespace Model
{
class ImageFileWriter : public IVideoOutput
{
   public:
    ImageFileWriter(const std::string& folder, const std::string& imageName);
    virtual ~ImageFileWriter();

    void open() override;
    void close() override;
    void writeImage(const Image& image) override;

   private:
    std::string folder_;
    std::string imageName_;

    Image image_;
    ImageConverter imageConverter_;
    HeapObjectFactory objectFactory_;
};

}    // namespace Model

#endif    // !COMPRESSED_IMAGE_FILE_WRITER_H
