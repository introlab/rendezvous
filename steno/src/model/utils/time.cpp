#include "time.h"

namespace Model
{
qint64 Time::milliseconds(const QString &hour, const QString &minute, const QString &second, const QString &millisecond)
{
    return hour.toLongLong() * 60 * 60 * 1000 + minute.toLongLong() * 60 * 1000 + second.toLongLong() * 1000 +
           millisecond.toLongLong();
}

}    // namespace Model
