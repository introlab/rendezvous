#ifndef I_STREAM_H
#define I_STREAM_H

#include <QObject>

namespace Model
{
enum class StreamStatus
{
    RUNNING,
    STOPPING,
    STOPPED,
    CRASHED
};

class IStream : public QObject
{
    Q_OBJECT

   public:
    virtual ~IStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual StreamStatus getStatus() const = 0;

   signals:
    void statusChanged();
};

}    // namespace Model

#endif    //! I_STREAM_H
