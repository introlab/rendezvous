#ifndef ZERO_COPY_CUDA_OBJECT_FACTORY_H
#define ZERO_COPY_CUDA_OBJECT_FACTORY_H

#include "model/stream/utils/alloc/i_object_factory.h"

namespace Model
{

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

} // Model

#endif //!ZERO_COPY_CUDA_OBJECT_FACTORY_H
