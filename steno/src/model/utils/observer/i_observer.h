#ifndef I_OBSERVER_H
#define I_OBSERVER_H

namespace Model
{
class IObserver
{
   public:
    virtual ~IObserver() = default;

    virtual void updateObserver() = 0;
};
}    // namespace Model

#endif    // I_OBSERVER_H
