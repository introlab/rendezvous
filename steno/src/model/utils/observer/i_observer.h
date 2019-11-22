#ifndef I_OBSERVER_H
#define I_OBSERVER_H

namespace Model
{
/**
 * @brief The IObserver class is the subscriber in the Observer design pattern. The one who gets notify.
 */
class IObserver
{
   public:
    virtual ~IObserver() = default;

    virtual void updateObserver() = 0;
};
}    // namespace Model

#endif    // I_OBSERVER_H
