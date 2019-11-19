#include "cuda_image_converter.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "model/stream/utils/images/image_format.h"
#include "model/stream/utils/math/cuda_helpers.cuh"

namespace Model
{
namespace
{
const int BLOCK_SIZE = 1024;

__device__ void getRGBFromUYVY(const UYVY& uyvy, RGB& rgb1, RGB& rgb2)
{
    int c1 = uyvy.y1 - 16;
    int c2 = uyvy.y2 - 16;
    int d = uyvy.u - 128;
    int e = uyvy.v - 128;

    rgb1.r = math::clamp((298 * c1 + 409 * e + 128) >> 8, 0, 255);
    rgb1.g = math::clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb1.b = math::clamp((298 * c1 + 516 * d + 128) >> 8, 0, 255);

    rgb2.r = math::clamp((298 * c2 + 409 * e + 128) >> 8, 0, 255);
    rgb2.g = math::clamp((298 * c2 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb2.b = math::clamp((298 * c2 + 516 * d + 128) >> 8, 0, 255);
}

__device__ void getRGBFromYUYV(const YUYV& yuyv, RGB& rgb1, RGB& rgb2)
{
    int c1 = yuyv.y1 - 16;
    int c2 = yuyv.y2 - 16;
    int d = yuyv.u - 128;
    int e = yuyv.v - 128;

    rgb1.r = math::clamp((298 * c1 + 409 * e + 128) >> 8, 0, 255);
    rgb1.g = math::clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb1.b = math::clamp((298 * c1 + 516 * d + 128) >> 8, 0, 255);

    rgb2.r = math::clamp((298 * c2 + 409 * e + 128) >> 8, 0, 255);
    rgb2.g = math::clamp((298 * c2 - 100 * d - 208 * e + 128) >> 8, 0, 255);
    rgb2.b = math::clamp((298 * c2 + 516 * d + 128) >> 8, 0, 255);
}

__device__ void getUYVYFromRGB(const RGB& rgb1, const RGB& rgb2, UYVY& uyvy)
{
    int r1 = (int)rgb1.r;
    int g1 = (int)rgb1.g;
    int b1 = (int)rgb1.b;
    int r2 = (int)rgb2.r;
    int g2 = (int)rgb2.g;
    int b2 = (int)rgb2.b;

    uyvy.u = ((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128;
    uyvy.y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    uyvy.v = ((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128;
    uyvy.y2 = ((66 * r2 + 129 * g2 + 25 * b2 + 128) >> 8) + 16;
}

__device__ void getYUYVFromRGB(const RGB& rgb1, const RGB& rgb2, YUYV& yuyv)
{
    int r1 = (int)rgb1.r;
    int g1 = (int)rgb1.g;
    int b1 = (int)rgb1.b;
    int r2 = (int)rgb2.r;
    int g2 = (int)rgb2.g;
    int b2 = (int)rgb2.b;

    yuyv.u = ((-38 * r1 - 74 * g1 + 112 * b1 + 128) >> 8) + 128;
    yuyv.y1 = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
    yuyv.v = ((112 * r1 - 94 * g1 - 18 * b1 + 128) >> 8) + 128;
    yuyv.y2 = ((66 * r2 + 129 * g2 + 25 * b2 + 128) >> 8) + 16;
}

__global__ void convertRGBToUYVYKernel(int size, const RGB* inData, UYVY* outData)
{
    int uyvyIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int rgbIndex = uyvyIndex * 2;

    if (uyvyIndex < size)
    {
        getUYVYFromRGB(inData[rgbIndex], inData[rgbIndex + 1], outData[uyvyIndex]);
    }
}

__global__ void convertRGBToYUYVKernel(int size, const RGB* inData, YUYV* outData)
{
    int yuyvIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int rgbIndex = yuyvIndex * 2;

    if (yuyvIndex < size)
    {
        getYUYVFromRGB(inData[rgbIndex], inData[rgbIndex + 1], outData[yuyvIndex]);
    }
}

__global__ void convertUYVYToRGBKernel(int size, const UYVY* inData, RGB* outData)
{
    int uyvyIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int rgbIndex = uyvyIndex * 2;

    if (uyvyIndex < size)
    {
        getRGBFromUYVY(inData[uyvyIndex], outData[rgbIndex], outData[rgbIndex + 1]);
    }
}

__global__ void convertYUYVToRGBKernel(int size, const YUYV* inData, RGB* outData)
{
    int yuyvIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int rgbIndex = yuyvIndex * 2;

    if (yuyvIndex < size)
    {
        getRGBFromYUYV(inData[yuyvIndex], outData[rgbIndex], outData[rgbIndex + 1]);
    }
}

}    // namespace

CudaImageConverter::CudaImageConverter(cudaStream_t stream)
    : stream_(stream)
{
}

void CudaImageConverter::convert(const Image& inImage, Image& outImage)
{
    int size = inImage.width * inImage.height;
    int blockCount = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (inImage.format == ImageFormat::RGB_FMT && outImage.format == ImageFormat::UYVY_FMT)
    {
        const RGB* rbgData = reinterpret_cast<const RGB*>(inImage.deviceData);
        UYVY* uyvyData = reinterpret_cast<UYVY*>(outImage.deviceData);

        convertRGBToUYVYKernel<<<blockCount / 2, BLOCK_SIZE, 0, stream_>>>(size / 2, rbgData, uyvyData);
    }
    else if (inImage.format == ImageFormat::RGB_FMT && outImage.format == ImageFormat::YUYV_FMT)
    {
        const RGB* rbgData = reinterpret_cast<const RGB*>(inImage.deviceData);
        YUYV* yuyvData = reinterpret_cast<YUYV*>(outImage.deviceData);

        convertRGBToYUYVKernel<<<blockCount / 2, BLOCK_SIZE, 0, stream_>>>(size / 2, rbgData, yuyvData);
    }
    else if (inImage.format == ImageFormat::UYVY_FMT && outImage.format == ImageFormat::RGB_FMT)
    {
        const UYVY* uyvyData = reinterpret_cast<const UYVY*>(inImage.deviceData);
        RGB* rbgData = reinterpret_cast<RGB*>(outImage.deviceData);

        convertUYVYToRGBKernel<<<blockCount / 2, BLOCK_SIZE, 0, stream_>>>(size / 2, uyvyData, rbgData);
    }
    else if (inImage.format == ImageFormat::YUYV_FMT && outImage.format == ImageFormat::RGB_FMT)
    {
        const YUYV* yuyvData = reinterpret_cast<const YUYV*>(inImage.deviceData);
        RGB* rbgData = reinterpret_cast<RGB*>(outImage.deviceData);

        convertYUYVToRGBKernel<<<blockCount / 2, BLOCK_SIZE, 0, stream_>>>(size / 2, yuyvData, rbgData);
    }
    else if (inImage.format == outImage.format)
    {
        std::cout << "Warning! Convertion input and output format are the same!" << std::endl;
        std::memcpy(outImage.hostData, inImage.hostData, inImage.size);
    }
    else
    {
        throw std::invalid_argument("Conversion from " + getImageFormatString(inImage.format) + " to " +
                                    getImageFormatString(outImage.format) + " is not defined!");
    }
    cudaStreamSynchronize(stream_);
    outImage.timeStamp = inImage.timeStamp;
}

}    // namespace Model