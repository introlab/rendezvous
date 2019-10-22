#ifndef TRIPLE_BUFFER_H
#define TRIPLE_BUFFER_H

template <typename T>
class TripleBuffer
{
public:

    template <typename... Args>
    TripleBuffer(Args&&... args)
        : first_(std::forward<Args>(args)...)
        , second_(std::forward<Args>(args)...)
        , third_(std::forward<Args>(args)...)
        , current_(&first_)
        , inUse1_(&second_)
        , inUse2_(&third_)
    {
    }

    T& getCurrent()
    {
        return *current_;
    }

    T& getInUse1()
    {
        return *inUse1_;
    }

    T& getInUse2()
    {
        return *inUse2_;
    }

    void swap()
    {
        std::swap(current_, inUse1_);
        std::swap(inUse1_, inUse2_);
    }

private:

    T first_;
    T second_;
    T third_;
    T* current_;
    T* inUse1_;
    T* inUse2_;

};

#endif // !TRIPLE_BUFFER_H
