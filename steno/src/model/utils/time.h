#ifndef TIME_H
#define TIME_H

#include <QString>

namespace Model
{
class Time
{
   public:
    static qint64 milliseconds(const QString &hour, const QString &minute, const QString &second,
                               const QString &millisecond);
    static QString clockFormat(const qint64 timeInSeconds);
};

}    // namespace Model

#endif    // TIME_H
