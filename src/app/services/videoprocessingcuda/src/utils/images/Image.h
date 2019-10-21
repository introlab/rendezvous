#ifndef IMAGE_H
#define IMAGE_H

#include <cstddef>

#include "utils/models/Dim3.h"

template<typename T>
struct ImageTemplate : public Dim3<int>
{
    ImageTemplate() = default;

    ImageTemplate(int width, int height, int channels)
        : Dim3(width, height, channels)
        , size(width * height * channels)
        , hostData(nullptr)
        , deviceData(nullptr)
    {
    }

    ImageTemplate(const Dim3& dim)
        : Dim3(dim)
        , size(dim.width * dim.height * dim.channels)
        , hostData(nullptr)
        , deviceData(nullptr)
    {
    }

    size_t size;
    
    T* hostData;
    T* deviceData;
};

struct Image : public ImageTemplate<unsigned char>
{
    Image() = default;

    Image(int width, int height, int channels)
        : ImageTemplate(width, height, channels)
    {
    }

    Image(const Dim3& dim)
        : ImageTemplate(dim)
    {
    }
};

struct ImageFloat : public ImageTemplate<float>
{
    ImageFloat() = default;

    ImageFloat(int width, int height, int channels)
        : ImageTemplate(width, height, channels)
    {
    }
    
    ImageFloat(const Dim3& dim)
        : ImageTemplate(dim)
    {
    }
};

#endif //!IMAGE_H
