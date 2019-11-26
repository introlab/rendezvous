#ifndef I_SUBJECT_H
#define I_SUBJECT_H

#include "model/utils/observer/i_observer.h"

namespace Model
{
/**
 * @brief The ISubject class is the emitter in the Observer design pattern.
 */
class ISubject
{
   public:
    virtual ~ISubject() = default;

    virtual void attach(IObserver* observer) = 0;
    virtual void notify() = 0;
    virtual void detach(IObserver* observer) = 0;
};
}    // namespace Model

#endif    // I_SUBJECT_H
