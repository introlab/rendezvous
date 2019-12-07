#include "subject.h"

namespace Model
{
/**
 * @brief Attach an object to the notifications send by Subject.
 * @param observer - object to attach to notifications.
 */
void Subject::attach(IObserver* observer)
{
    if (observer != nullptr)
    {
        m_subscribers.push_back(observer);
    }
}

/**
 * @brief Remove an observer from the list of subscribers.
 * @param observer - object to remove from the notifications list.
 */
void Subject::detach(IObserver* observer)
{
    for (int index = 0; static_cast<size_t>(index) < m_subscribers.size(); ++index)
    {
        if (m_subscribers.at(static_cast<size_t>(index)) == observer)
        {
            m_subscribers.erase(m_subscribers.begin() + index);
        }
    }
}

/**
 * @brief Notify all observers that the state of Subject changed.
 */
void Subject::notify()
{
    for (auto observer : m_subscribers)
    {
        observer->updateObserver();
    }
}
}    // namespace Model
