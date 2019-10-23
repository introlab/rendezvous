#include "ZeroCopyCudaObjectFactory.h"

#include "utils/CudaUtils.cuh"

namespace
{
    template <typename T>
    void mallocHost(T*& hostPtr, std::size_t size)
    {
        checkCuda(cudaHostAlloc(&hostPtr,  size * sizeof(T),  cudaHostAllocMapped));
    }

    template <typename T>
    void deallocHost(T*& hostPtr)
    {
        checkCuda(cudaFreeHost(hostPtr));
        hostPtr = nullptr;
    }

    template <typename T>
    void getDevicePointer(T* hostPtr, T*& devicePtr)
    {
        checkCuda(cudaHostGetDevicePointer(&devicePtr, hostPtr, 0));
    }
}

ZeroCopyCudaObjectFactory::ZeroCopyCudaObjectFactory()
{
    cudaSetDeviceFlags(cudaDeviceMapHost);
}

void ZeroCopyCudaObjectFactory::allocateObject(Image& image) const
{
    mallocHost(image.hostData, image.size);
    getDevicePointer(image.hostData, image.deviceData);
}

void ZeroCopyCudaObjectFactory::deallocateObject(Image& image) const
{
    deallocHost(image.hostData);
    image.deviceData = nullptr;
}

void ZeroCopyCudaObjectFactory::allocateObject(ImageFloat& image) const
{
    mallocHost(image.hostData, image.size);
    getDevicePointer(image.hostData, image.deviceData);
}

void ZeroCopyCudaObjectFactory::deallocateObject(ImageFloat& image) const
{
    deallocHost(image.hostData);
    image.deviceData = nullptr;
}

void ZeroCopyCudaObjectFactory::allocateObject(DewarpingMapping& mapping) const
{
    mallocHost(mapping.hostData, mapping.size);
    getDevicePointer(mapping.hostData, mapping.deviceData);
}

void ZeroCopyCudaObjectFactory::deallocateObject(DewarpingMapping& mapping) const
{
    deallocHost(mapping.hostData);
    mapping.deviceData = nullptr;
}

void ZeroCopyCudaObjectFactory::allocateObject(FilteredDewarpingMapping& mapping) const
{
    mallocHost(mapping.hostData, mapping.size);
    getDevicePointer(mapping.hostData, mapping.deviceData);
}

void ZeroCopyCudaObjectFactory::deallocateObject(FilteredDewarpingMapping& mapping) const
{
    deallocHost(mapping.hostData);
    mapping.deviceData = nullptr;
}