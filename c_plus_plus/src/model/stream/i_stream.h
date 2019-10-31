#ifndef I_STREAM_H
#define I_STREAM_H

#include <QObject>

namespace Model
{
class IStream : public QObject
{
   public:
    virtual ~IStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
};

}    // namespace Model

#endif    //! I_STREAM_H
