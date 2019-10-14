#ifndef MANAGED_MEMORY_CUDA_OBJECT_FACTORY_H
#define MANAGED_MEMORY_CUDA_OBJECT_FACTORY_H

#include <cuda_runtime.h>

#include "utils/objects/IObjectFactory.h"

class ManagedMemoryCudaObjectFactory : public IObjectFactory
{
public:

    ManagedMemoryCudaObjectFactory(const cudaStream_t& stream);

    void allocateObject(Image& cudaImage) const override;
    void deallocateObject(Image& cudaImage) const override;

    void allocateObject(ImageFloat& cudaImage) const override;
    void deallocateObject(ImageFloat& cudaImage) const override;

    void allocateObject(DewarpingMapping& dewarpingMapping) const override;
    void deallocateObject(DewarpingMapping& dewarpingMapping) const override;

    void allocateObject(FilteredDewarpingMapping& mapping) const override;
    void deallocateObject(FilteredDewarpingMapping& mapping) const override;

private:

    cudaStream_t stream_;
    
};

#endif //!MANAGED_MEMORY_CUDA_OBJECT_FACTORY_H