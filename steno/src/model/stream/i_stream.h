#ifndef I_STREAM_H
#define I_STREAM_H

#include <QObject>

namespace Model
{
class IStream : public QObject
{
    Q_OBJECT

   public:
    enum State
    {
        Started,
        Stopping,
        Stopped
    };

    virtual ~IStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual State state() const = 0;

   signals:
    void stateChanged(const State& state);
};

}    // namespace Model

#endif    //! I_STREAM_H
