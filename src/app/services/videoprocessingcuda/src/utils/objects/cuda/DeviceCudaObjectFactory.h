#ifndef DEVICE_CUDA_OBJECT_FACTORY_H
#define DEVICE_CUDA_OBJECT_FACTORY_H

#include "utils/objects/IObjectFactory.h"

class DeviceCudaObjectFactory : public IObjectFactory
{
public:

    void allocateObject(Image& cudaImage) const override;
    void deallocateObject(Image& cudaImage) const override;
    void allocateObject(ImageFloat& cudaImage) const override;
    void deallocateObject(ImageFloat& cudaImage) const override;
    void allocateObject(DewarpingMapping& dewarpingMapping) const override;
    void deallocateObject(DewarpingMapping& dewarpingMapping) const override;
    void allocateObject(FilteredDewarpingMapping& mapping) const override;
    void deallocateObject(FilteredDewarpingMapping& mapping) const override;
};

#endif //!DEVICE_CUDA_OBJECT_FACTORY_H