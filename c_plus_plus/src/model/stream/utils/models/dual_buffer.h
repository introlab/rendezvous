#ifndef DUAL_BUFFER_H
#define DUAL_BUFFER_H

namespace Model
{
template <typename T>
class DualBuffer
{
   public:
    template <typename... Args>
    DualBuffer(Args&&... args)
        : first_(std::forward<Args>(args)...)
        , second_(std::forward<Args>(args)...)
        , current_(&first_)
        , inUse_(&second_)
    {
    }

    T& getCurrent() { return *current_; }

    T& getInUse() { return *inUse_; }

    void swap() { std::swap(current_, inUse_); }

   private:
    T first_;
    T second_;
    T* current_;
    T* inUse_;
};

}    // namespace Model

#endif    // !DUAL_BUFFER_H
