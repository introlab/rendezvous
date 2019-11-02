#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <vector>

namespace Model
{
template <typename T>
class CircularBuffer
{
   public:
    explicit CircularBuffer(std::size_t size, const T& value = T())
        : size_(size)
        , buffers_(size, value)
        , index_(0)
    {
    }

    std::size_t size()
    {
        return size_;
    }

    T& current()
    {
        return buffers_[index_];
    }

    void next()
    {
        index_ = (index_ + 1) % size_;
    }

   private:
    std::size_t size_;
    std::vector<T> buffers_;
    int index_;
};

}    // namespace Model

#endif    //! CIRCULAR_BUFFER_H
