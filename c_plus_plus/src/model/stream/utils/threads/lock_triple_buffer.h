#ifndef LOCK_TRIPLER_BUFFER
#define LOCK_TRIPLER_BUFFER

#include <array>
#include <atomic>
#include <iostream>
#include <vector>

namespace Model
{

template<typename T>
class LockTripleBuffer
{
public:

    explicit LockTripleBuffer(const T& value = T())
        : buf_(3, value)
        , current_(&buf_[0])
        , inUse_(&buf_[1])
        , free_(&buf_[2])
        , locked_(&buf_[2])
        , swapCounter_(0)
    {
    }

    T& getCurrent()
    {
        getLock();
        T* current = current_;
        releaseLock();
        return *current;
    }

    T& getInUse()
    {
        getLock();
        T* inUse = inUse_;
        releaseLock();
        return *inUse;
    }

    T& getFree()
    {
        getLock();
        T* free = free_;
        releaseLock();
        return *free;
    }

    T& getLocked()
    {
        getLock();
        T* locked = locked_;
        releaseLock();
        return *locked;
    }

    void swap()
    {
        getLock();
        if (inUse_ == locked_)
        {
            std::swap(inUse_, free_);
        }
        std::swap(current_, inUse_);
        ++swapCounter_;
        releaseLock();
    }

    void lockInUse()
    {
        getLock();
        locked_ = inUse_;
        releaseLock();
    }

    int getAndClearSwapCount()
    {
        getLock();
        int swapCounter = swapCounter_;
        swapCounter_ = 0;
        releaseLock();
        return swapCounter;
    }

private:

    void getLock()
    {
        while (lock_.test_and_set(std::memory_order_acquire)) {}
    }

    void releaseLock()
    {
        lock_.clear(std::memory_order_release);
    }

    std::vector<T> buf_;
    T* current_;
    T* inUse_;
    T* free_;
    T* locked_;
    int swapCounter_;
    std::atomic_flag lock_;

};

} // Model

#endif //!LOCK_TRIPLER_BUFFER
