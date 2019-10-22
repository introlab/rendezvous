#ifndef CUDA_IMAGE_CONVERTER_H
#define CUDA_IMAGE_CONVERTER_H

#include "utils/images/IImageConverter.h"

class CudaImageConverter : public IImageConverter
{
public:

    void convert(const Image& inImage, const Image& outImage) override;

};

#endif //!CUDA_IMAGE_CONVERTER_H