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
}

#endif    // I_OBSERVER_H
