#ifndef SUBJECT_H
#define SUBJECT_H

#include "model/utils/observer/i_observer.h"

#include <vector>

namespace Model
{
/**
 * @brief The Subject class is the emitter in the Observer design pattern.
 */
class Subject
{
   public:
    void attach(IObserver* observer);
    void notify();
    void detach(IObserver* observer);

   private:
    std::vector<IObserver*> m_subscribers;
};
}    // namespace Model

#endif    // SUBJECT_H
