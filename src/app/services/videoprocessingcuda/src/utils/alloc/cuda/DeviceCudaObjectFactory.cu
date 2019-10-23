#include "DeviceCudaObjectFactory.h"

#include "utils/CudaUtils.cuh"

namespace
{
    template <typename T>
    void mallocDevice(T*& ptr, std::size_t size)
    {
        checkCuda(cudaMalloc(&ptr, size * sizeof(T)));
    }

    template <typename T>
    void deallocManaged(T*& ptr)
    {
        checkCuda(cudaFree(ptr));
        ptr = nullptr;
    }
}

void DeviceCudaObjectFactory::allocateObject(ImageFloat& image) const
{
    mallocDevice(image.deviceData, image.size);
}

void DeviceCudaObjectFactory::deallocateObject(ImageFloat& image) const
{
    deallocManaged(image.deviceData);
}

void DeviceCudaObjectFactory::allocateObject(Image& image) const
{
    mallocDevice(image.deviceData, image.size);
}

void DeviceCudaObjectFactory::deallocateObject(Image& image) const
{
    deallocManaged(image.deviceData);
}

void DeviceCudaObjectFactory::allocateObject(DewarpingMapping& mapping) const
{
    mallocDevice(mapping.deviceData, mapping.size);
}

void DeviceCudaObjectFactory::deallocateObject(DewarpingMapping& mapping) const
{
    deallocManaged(mapping.deviceData);
}

void DeviceCudaObjectFactory::allocateObject(FilteredDewarpingMapping& mapping) const
{
    mallocDevice(mapping.deviceData, mapping.size);
}

void DeviceCudaObjectFactory::deallocateObject(FilteredDewarpingMapping& mapping) const
{
    deallocManaged(mapping.deviceData);
}