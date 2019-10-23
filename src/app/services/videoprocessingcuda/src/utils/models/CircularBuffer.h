#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <memory>

template <typename T>
class CircularBuffer
{
public:

    CircularBuffer(std::size_t size, const T& value = T())
        : size_(size)
        , buffers_(std::make_unique<T[]>(size))
        , index_(0)
    {
        for (std::size_t i = 0; i < size_; ++i)
        {
            buffers_[i] = value;
        }
    }

    std::size_t size()
    {
        return size_;
    }

    const std::unique_ptr<T[]>& buffers()
    {
        return buffers_;
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
    std::unique_ptr<T[]> buffers_;
    int index_;

};

#endif //!CIRCULAR_BUFFER_H