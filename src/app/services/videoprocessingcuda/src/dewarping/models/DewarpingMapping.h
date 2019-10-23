#ifndef DEWAPING_MAPPING_H
#define DEWAPING_MAPPING_H

#include <cstddef>

#include "dewarping/models/LinearPixelFilter.h"
#include "utils/models/Dim2.h"

template<typename T>
struct DewarpingMappingTemplate : public Dim2<int>
{
    DewarpingMappingTemplate() = default;

    DewarpingMappingTemplate(int width, int height)
        : Dim2(width, height)
        , size(width * height)
        , hostData(nullptr)
        , deviceData(nullptr)
    {
    }

    DewarpingMappingTemplate(const Dim2& dim)
        : Dim2(dim)
        , size(dim.width * dim.height)
        , hostData(nullptr)
        , deviceData(nullptr)
    {
    }

    std::size_t size;

    T* hostData;
    T* deviceData;

};

struct DewarpingMapping : DewarpingMappingTemplate<int>
{
    DewarpingMapping() = default;

    DewarpingMapping(int width, int height)
        : DewarpingMappingTemplate(width, height)
    {
    }

    DewarpingMapping(const Dim2& dim)
        : DewarpingMappingTemplate(dim)
    {
    }
};

struct FilteredDewarpingMapping : DewarpingMappingTemplate<LinearPixelFilter>
{
    FilteredDewarpingMapping() = default;

    FilteredDewarpingMapping(int width, int height)
        : DewarpingMappingTemplate(width, height)
    {
    }

    FilteredDewarpingMapping(const Dim2& dim)
        : DewarpingMappingTemplate(dim)
    {
    }
};

#endif // !DEWAPING_MAPPING_H