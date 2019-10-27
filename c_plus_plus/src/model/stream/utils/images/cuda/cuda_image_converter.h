#ifndef CUDA_IMAGE_CONVERTER_H
#define CUDA_IMAGE_CONVERTER_H

#include <cuda_runtime.h>

#include "model/stream/utils/images/i_image_converter.h"

namespace Model
{

class CudaImageConverter : public IImageConverter
{
public:

    explicit CudaImageConverter(cudaStream_t stream);

    void convert(const Image& inImage, const Image& outImage) override;

private:

    cudaStream_t stream_;

};

} // Model

#endif //!CUDA_IMAGE_CONVERTER_H
