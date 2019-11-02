#ifndef DEVICE_CUDA_OBJECT_FACTORY_H
#define DEVICE_CUDA_OBJECT_FACTORY_H

#include "model/stream/utils/alloc/i_object_factory.h"

namespace Model
{
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

}    // namespace Model

#endif    //! DEVICE_CUDA_OBJECT_FACTORY_H
