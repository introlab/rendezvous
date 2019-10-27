#include "managed_memory_cuda_object_factory.h"

#include "model/stream/utils/cuda_utils.cuh"

namespace
{
    template <typename T>
    void mallocManaged(T*& ptr, std::size_t size, const cudaStream_t& stream)
    {
        checkCuda(cudaMallocManaged(&ptr, size * sizeof(T)));
        checkCuda(cudaStreamAttachMemAsync(stream, ptr));
        cudaDeviceSynchronize();
    }

    template <typename T>
    void deallocManaged(T*& ptr)
    {
        checkCuda(cudaFree(ptr));
        ptr = nullptr;
    }
}

ManagedMemoryCudaObjectFactory::ManagedMemoryCudaObjectFactory(const cudaStream_t& stream)
    : stream_(stream)
{
}

void ManagedMemoryCudaObjectFactory::allocateObject(ImageFloat& image) const
{
    mallocManaged(image.hostData, image.size, stream_);
    image.deviceData = image.hostData; // With managed memory host and device memory is same ptr
}

void ManagedMemoryCudaObjectFactory::deallocateObject(ImageFloat& image) const
{
    deallocManaged(image.hostData);
    image.deviceData = nullptr; // With managed memory host and device memory is same ptr (so only dealloc once)
}

void ManagedMemoryCudaObjectFactory::allocateObject(Image& image) const
{
    mallocManaged(image.hostData, image.size, stream_);
    image.deviceData = image.hostData; // With managed memory host and device memory is same ptr
}

void ManagedMemoryCudaObjectFactory::deallocateObject(Image& image) const
{
    deallocManaged(image.hostData);
    image.deviceData = nullptr; // With managed memory host and device memory is same ptr (so only dealloc once)
}

void ManagedMemoryCudaObjectFactory::allocateObject(DewarpingMapping& mapping) const
{
    mallocManaged(mapping.hostData, mapping.size, stream_);
    mapping.deviceData = mapping.hostData; // With managed memory host and device memory is same ptr
}

void ManagedMemoryCudaObjectFactory::deallocateObject(DewarpingMapping& mapping) const
{
    deallocManaged(mapping.hostData);
    mapping.deviceData = nullptr; // With managed memory host and device memory is same ptr (so only dealloc once)
}

void ManagedMemoryCudaObjectFactory::allocateObject(FilteredDewarpingMapping& mapping) const
{
    mallocManaged(mapping.hostData, mapping.size, stream_);
    mapping.deviceData = mapping.hostData; // With managed memory host and device memory is same ptr
}

void ManagedMemoryCudaObjectFactory::deallocateObject(FilteredDewarpingMapping& mapping) const
{
    deallocManaged(mapping.hostData);
    mapping.deviceData = nullptr; // With managed memory host and device memory is same ptr (so only dealloc once)
}