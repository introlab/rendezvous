#ifndef ZERO_COPY_CUDA_OBJECT_FACTORY_H
#define ZERO_COPY_CUDA_OBJECT_FACTORY_H

#include "utils/alloc/IObjectFactory.h"

class ZeroCopyCudaObjectFactory : public IObjectFactory
{
public:

    ZeroCopyCudaObjectFactory();

    void allocateObject(Image& image) const override;
    void deallocateObject(Image& image) const override;

    void allocateObject(ImageFloat& image) const override;
    void deallocateObject(ImageFloat& image) const override;

    void allocateObject(DewarpingMapping& mapping) const override;
    void deallocateObject(DewarpingMapping& mapping) const override;

    void allocateObject(FilteredDewarpingMapping& mapping) const override;
    void deallocateObject(FilteredDewarpingMapping& mapping) const override;
};

#endif //!ZERO_COPY_CUDA_OBJECT_FACTORY_H