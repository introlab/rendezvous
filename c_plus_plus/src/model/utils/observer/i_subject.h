#ifndef I_SUBJECT_H
#define I_SUBJECT_H

#include "model/utils/observer/i_observer.h"

namespace Model
{
class ISubject
{
   public:
    virtual ~ISubject() = default;

    virtual void attach(IObserver* observer) = 0;
    virtual void notify() = 0;
};
}

#endif    // I_SUBJECT_H
