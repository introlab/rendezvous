#ifndef I_STREAM_H
#define I_STREAM_H

#include <QWidget>

namespace Model
{
class IStream : public QWidget
{
   Q_OBJECT

   public:
    enum State {
        Started,
        Stopped
    };

    virtual ~IStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual IStream::State state() const = 0;

   signals:
    void stateChanged(const IStream::State& state);
};

}    // namespace Model

#endif    //! I_STREAM_H
