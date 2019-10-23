#ifndef CUDA_IMAGE_CONVERTER_H
#define CUDA_IMAGE_CONVERTER_H

#include <cuda_runtime.h>

#include "utils/images/IImageConverter.h"

class CudaImageConverter : public IImageConverter
{
public:

    CudaImageConverter(cudaStream_t stream);

    void convert(const Image& inImage, const Image& outImage) override;

private:

    cudaStream_t stream_;

};

#endif //!CUDA_IMAGE_CONVERTER_H