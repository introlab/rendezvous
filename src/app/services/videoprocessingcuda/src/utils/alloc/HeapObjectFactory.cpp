#include "HeapObjectFactory.h"

namespace
{
    template <typename T>
    void malloc(T*& ptr, std::size_t size)
    {
        ptr = new T[size];
    }

    template <typename T>
    void dealloc(T*& ptr)
    {
        delete[] ptr;
        ptr = nullptr;
    }
}

void HeapObjectFactory::allocateObject(Image& image) const
{
    malloc(image.hostData, image.size);
}

void HeapObjectFactory::deallocateObject(Image& image) const
{
    dealloc(image.hostData);
}

void HeapObjectFactory::allocateObject(ImageFloat& image) const
{
    malloc(image.hostData, image.size);
}

void HeapObjectFactory::deallocateObject(ImageFloat& image) const
{
    dealloc(image.hostData);
}

void HeapObjectFactory::allocateObject(DewarpingMapping& mapping) const
{
    malloc(mapping.hostData, mapping.size);
}

void HeapObjectFactory::deallocateObject(DewarpingMapping& mapping) const
{
    dealloc(mapping.hostData);
}

void HeapObjectFactory::allocateObject(FilteredDewarpingMapping& mapping) const
{
    malloc(mapping.hostData, mapping.size);
}

void HeapObjectFactory::deallocateObject(FilteredDewarpingMapping& mapping) const
{
    dealloc(mapping.hostData);
}