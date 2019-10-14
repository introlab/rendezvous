#ifndef HEAP_OBJECT_FACTORY_H
#define HEAP_OBJECT_FACTORY_H

#include "utils/objects/IObjectFactory.h"

class HeapObjectFactory : public IObjectFactory
{
public:

    void allocateObject(Image& image) const override;
    void deallocateObject(Image& image) const override;

    void allocateObject(ImageFloat& image) const override;
    void deallocateObject(ImageFloat& image) const override;

    void allocateObject(DewarpingMapping& mapping) const override;
    void deallocateObject(DewarpingMapping& mapping) const override;

    void allocateObject(FilteredDewarpingMapping& mapping) const override;
    void deallocateObject(FilteredDewarpingMapping& mapping) const override;
};

#endif //!HEAP_OBJECT_FACTORY_H