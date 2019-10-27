#ifndef I_OBJECT_FACTORY_H
#define I_OBJECT_FACTORY_H

#include <vector>

#include "model/stream/utils/images/images.h"
#include "model/stream/utils/models/dual_buffer.h"
#include "model/stream/utils/threads/lock_triple_buffer.h"
#include "model/stream/video/dewarping/models/dewarping_mapping.h"

namespace Model
{
class IObjectFactory
{
   public:
    virtual ~IObjectFactory() = default;
    virtual void allocateObject(Image& image) const = 0;
    virtual void deallocateObject(Image& image) const = 0;
    virtual void allocateObject(ImageFloat& image) const = 0;
    virtual void deallocateObject(ImageFloat& image) const = 0;
    virtual void allocateObject(DewarpingMapping& mapping) const = 0;
    virtual void deallocateObject(DewarpingMapping& mapping) const = 0;
    virtual void allocateObject(FilteredDewarpingMapping& mapping) const = 0;
    virtual void deallocateObject(FilteredDewarpingMapping& mapping) const = 0;

    template <typename T>
    void allocateObjectDualBuffer(DualBuffer<T>& buffer) const
    {
        allocateObject(buffer.getCurrent());
        allocateObject(buffer.getInUse());
    }

    template <typename T>
    void deallocateObjectDualBuffer(DualBuffer<T>& buffer) const
    {
        deallocateObject(buffer.getCurrent());
        deallocateObject(buffer.getInUse());
    }

    template <typename T>
    void allocateObjectLockTripleBuffer(LockTripleBuffer<T>& buffer) const
    {
        allocateObject(buffer.getCurrent());
        allocateObject(buffer.getInUse());
        allocateObject(buffer.getFree());
    }

    template <typename T>
    void deallocateObjectLockTripleBuffer(LockTripleBuffer<T>& buffer) const
    {
        deallocateObject(buffer.getCurrent());
        deallocateObject(buffer.getInUse());
        deallocateObject(buffer.getFree());
    }

    template <typename T>
    void allocateObjectVector(std::vector<T>& vector) const
    {
        for (T& element : vector)
        {
            allocateObject(element);
        }
    }

    template <typename T>
    void deallocateObjectVector(std::vector<T>& vector) const
    {
        for (T& element : vector)
        {
            deallocateObject(element);
        }
    }
};

}    // namespace Model

#endif    //! I_OBJECT_FACTORY_H
