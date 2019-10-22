#ifndef IMAGE_FILE_WRITER_H
#define IMAGE_FILE_WRITER_H

#include <memory>

#include "stream/output/IVideoOutput.h"
#include "utils/images/ImageConverter.h"
#include "utils/alloc/HeapObjectFactory.h"

class ImageFileWriter : public IVideoOutput
{
public:

    ImageFileWriter(const std::string& folder, const std::string& imageName);
    virtual ~ImageFileWriter();

    void writeImage(const Image& image) override;
    
private:

    std::string folder_;
    std::string imageName_;

    Image image_;
    ImageConverter imageConverter_;
    HeapObjectFactory objectFactory_;

};

#endif // !COMPRESSED_IMAGE_FILE_WRITER_H
